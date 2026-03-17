"""
Verification script for the staged join template change.

Run with: python test_staged_join_template.py

This script:
1. Renders the template with <=15 feature views and verifies the SQL is identical
   to what the old (single-join) template would produce.
2. Renders the template with >15 feature views and verifies structural correctness
   (correct number of staged temp tables, final SELECT has all columns, etc.)
3. Provides a Parquet diff utility to compare outputs between old and new runs.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, KeysView
import re
import sys

from jinja2 import BaseLoader, Environment

from sdk.python.feast.infra.offline_stores.bigquery import (
    MULTIPLE_FEATURE_VIEW_POINT_IN_TIME_JOIN,
)
from sdk.python.feast.infra.offline_stores.offline_utils import (
    FeatureViewQueryContext,
    build_point_in_time_query,
)


# The old template (before staged joins) for comparison
OLD_FINAL_SECTION = """
SELECT {{ final_output_feature_names | backticks | join(', ')}}
FROM entity_dataframe
{% for featureview in featureviews %}
LEFT JOIN (
    SELECT * EXCEPT ( {{ featureview.entities | join(', ') }})
    FROM {{ featureview.name }}__cleaned
) USING ({{featureview.name}}__entity_row_unique_id)
{% endfor %}
"""


def make_mock_fv(i: int) -> FeatureViewQueryContext:
    """Create a mock FeatureViewQueryContext for testing."""
    return FeatureViewQueryContext(
        name=f"fv_{i}",
        ttl=3600,
        entities=["entity_id"],
        features=[f"feature_{i}_a", f"feature_{i}_b"],
        field_mapping={},
        timestamp_field="event_timestamp",
        created_timestamp_column=None,
        table_subquery=f"`project.dataset.fv_{i}_table`",
        entity_selections=["entity_id"],
        min_event_timestamp="2024-01-01",
        max_event_timestamp="2024-12-31",
        date_partition_column=None,
    )


def build_old_template() -> str:
    """Reconstruct the old template by replacing the final section."""
    # Everything up to the final join section is the same
    # Find the staged join comment and replace everything after it
    marker = "/*\n Joins the outputs of multiple time travel joins"
    parts = MULTIPLE_FEATURE_VIEW_POINT_IN_TIME_JOIN.split(marker)
    return parts[0] + "/*\n Joins the outputs of multiple time travel joins to a single table.\n The entity_dataframe dataset being our source of truth here.\n */" + OLD_FINAL_SECTION


def render_query(template_str: str, fv_contexts: list, entity_df_columns: list) -> str:
    env = Environment(loader=BaseLoader())
    env.filters["backticks"] = lambda x: f"`{x}`" if isinstance(x, str) else [f"`{v}`" for v in x]
    template = env.from_string(source=template_str)

    final_output_feature_names = list(entity_df_columns)
    for fv in fv_contexts:
        for feature in fv.features:
            final_output_feature_names.append(
                fv.field_mapping.get(feature, feature)
            )

    context = {
        "left_table_query_string": "project.dataset.entity_table",
        "entity_df_event_timestamp_col": "event_timestamp",
        "unique_entity_keys": set(e for fv in fv_contexts for e in fv.entities),
        "featureviews": [asdict(ctx) for ctx in fv_contexts],
        "full_feature_names": False,
        "final_output_feature_names": final_output_feature_names,
    }
    return template.render(context)


def normalize_sql(sql: str) -> str:
    """Normalize whitespace and remove comments for comparison."""
    # Remove block comments
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    # Remove single-line comments
    sql = re.sub(r"--[^\n]*", "", sql)
    # Collapse whitespace
    sql = re.sub(r"\s+", " ", sql).strip()
    # Normalize spaces inside parens: "( x)" -> "(x)", "( x )" -> "(x)"
    sql = re.sub(r"\(\s+", "(", sql)
    sql = re.sub(r"\s+\)", ")", sql)
    return sql


def test_small_case_identical():
    """With <=15 feature views, the output should be identical to the old template."""
    print("Test 1: <=15 feature views produces identical SQL... ", end="")

    for n_fvs in [1, 5, 10, 15]:
        fvs = [make_mock_fv(i) for i in range(n_fvs)]
        entity_cols = ["entity_id", "event_timestamp"]

        old_template = build_old_template()
        new_query = render_query(MULTIPLE_FEATURE_VIEW_POINT_IN_TIME_JOIN, fvs, entity_cols)
        old_query = render_query(old_template, fvs, entity_cols)

        if normalize_sql(new_query) != normalize_sql(old_query):
            print(f"FAILED (n_fvs={n_fvs})")
            # Show the diff in the final section
            new_final = new_query.split("Joins the outputs")[1]
            old_final = old_query.split("Joins the outputs")[1]
            print(f"  OLD final section:\n{old_final[:500]}")
            print(f"  NEW final section:\n{new_final[:500]}")
            return False

    print("PASSED")
    return True


def test_large_case_structure():
    """With >15 feature views, verify structural correctness."""
    print("Test 2: >15 feature views produces correct staged structure... ", end="")

    for n_fvs in [16, 30, 45, 55, 60]:
        fvs = [make_mock_fv(i) for i in range(n_fvs)]
        entity_cols = ["entity_id", "event_timestamp"]

        query = render_query(MULTIPLE_FEATURE_VIEW_POINT_IN_TIME_JOIN, fvs, entity_cols)

        # Count staged temp tables
        n_staged = query.count("CREATE TEMP TABLE __staged_join_")
        expected_batches = (n_fvs + 14) // 15  # ceil division
        expected_staged = expected_batches - 1  # last batch is a SELECT, not CREATE TEMP TABLE

        if n_staged != expected_staged:
            print(f"FAILED (n_fvs={n_fvs}): expected {expected_staged} staged tables, got {n_staged}")
            return False

        # Verify final SELECT uses final_output_feature_names (has backtick-quoted columns)
        # The last statement should be a SELECT, not a CREATE TEMP TABLE
        statements = [s.strip() for s in query.split(";") if s.strip()]
        last_stmt = statements[-1]
        if not last_stmt.strip().startswith("SELECT"):
            print(f"FAILED (n_fvs={n_fvs}): last statement is not a SELECT")
            print(f"  Last statement starts with: {last_stmt[:100]}")
            return False

        # Verify all feature view cleaned tables are referenced
        for i in range(n_fvs):
            if f"fv_{i}__cleaned" not in query:
                print(f"FAILED (n_fvs={n_fvs}): fv_{i}__cleaned not found in query")
                return False

        # Verify all feature columns appear in the final SELECT
        for i in range(n_fvs):
            for col in ["feature_{}_a".format(i), "feature_{}_b".format(i)]:
                if f"`{col}`" not in last_stmt:
                    print(f"FAILED (n_fvs={n_fvs}): {col} not in final SELECT")
                    return False

    print("PASSED")
    return True


def test_staged_join_references():
    """Verify that each staged join references the previous one correctly."""
    print("Test 3: staged joins chain correctly... ", end="")

    fvs = [make_mock_fv(i) for i in range(55)]
    entity_cols = ["entity_id", "event_timestamp"]
    query = render_query(MULTIPLE_FEATURE_VIEW_POINT_IN_TIME_JOIN, fvs, entity_cols)

    statements = [s.strip() for s in query.split(";") if s.strip()]

    # Find the staged join statements (after all the __cleaned tables)
    staged_stmts = [s for s in statements if "__staged_join_" in s or "final_output" in s.lower()]

    # First staged join should reference entity_dataframe
    first_staged = [s for s in statements if "__staged_join_0" in s and "CREATE TEMP TABLE" in s]
    if not first_staged:
        print("FAILED: no __staged_join_0 found")
        return False
    if "FROM entity_dataframe" not in first_staged[0]:
        print("FAILED: __staged_join_0 doesn't reference entity_dataframe")
        return False

    # Middle staged joins should reference previous staged join
    for i in range(1, 3):  # We expect __staged_join_1 and __staged_join_2 for 55 FVs
        staged = [s for s in statements if f"__staged_join_{i}" in s and "CREATE TEMP TABLE" in s]
        if not staged:
            print(f"FAILED: no __staged_join_{i} found")
            return False
        if f"FROM __staged_join_{i-1}" not in staged[0]:
            print(f"FAILED: __staged_join_{i} doesn't reference __staged_join_{i-1}")
            return False

    # Final SELECT should reference the last staged join
    last_stmt = statements[-1]
    if "FROM __staged_join_" not in last_stmt:
        print("FAILED: final SELECT doesn't reference a staged join")
        return False

    print("PASSED")
    return True


def compare_parquet_outputs(old_dir: str, new_dir: str):
    """Compare Parquet outputs from two pipeline runs.

    Usage:
        python test_staged_join_template.py compare gs://bucket/old_output gs://bucket/new_output
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas is required for Parquet comparison: pip install pandas")
        sys.exit(1)

    try:
        import pyarrow.parquet as pq
        import gcsfs
    except ImportError:
        print("pyarrow and gcsfs are required: pip install pyarrow gcsfs")
        sys.exit(1)

    fs = gcsfs.GCSFileSystem()

    def read_all_parquet(path: str) -> pd.DataFrame:
        files = fs.glob(f"{path.removeprefix('gs://')}/*.parquet")
        if not files:
            files = fs.glob(f"{path.removeprefix('gs://')}/**/*.parquet")
        dfs = [pd.read_parquet(f"gs://{f}") for f in files]
        if not dfs:
            print(f"No parquet files found under {path}")
            sys.exit(1)
        return pd.concat(dfs, ignore_index=True)

    print(f"Reading old output from {old_dir}...")
    old_df = read_all_parquet(old_dir)
    print(f"  Shape: {old_df.shape}")

    print(f"Reading new output from {new_dir}...")
    new_df = read_all_parquet(new_dir)
    print(f"  Shape: {new_df.shape}")

    # Compare schemas
    old_cols = set(old_df.columns)
    new_cols = set(new_df.columns)

    if old_cols != new_cols:
        print("\nCOLUMN MISMATCH:")
        missing = old_cols - new_cols
        extra = new_cols - old_cols
        if missing:
            print(f"  Missing in new: {missing}")
        if extra:
            print(f"  Extra in new: {extra}")
    else:
        print(f"\nColumns match ({len(old_cols)} columns)")

    # Compare row counts
    if len(old_df) != len(new_df):
        print(f"\nROW COUNT MISMATCH: old={len(old_df)}, new={len(new_df)}")
    else:
        print(f"Row counts match ({len(old_df)} rows)")

    # Sort both by the same columns for comparison
    common_cols = sorted(old_cols & new_cols)
    sort_cols = [c for c in common_cols if c in ("entity_id", "event_timestamp")]
    if sort_cols:
        old_df = old_df.sort_values(sort_cols).reset_index(drop=True)
        new_df = new_df.sort_values(sort_cols).reset_index(drop=True)

    # Compare values column by column
    mismatches = []
    for col in common_cols:
        if col not in old_df.columns or col not in new_df.columns:
            continue
        try:
            if not old_df[col].equals(new_df[col]):
                n_diff = (old_df[col] != new_df[col]).sum()
                mismatches.append((col, n_diff))
        except Exception:
            mismatches.append((col, "comparison error"))

    if mismatches:
        print(f"\nVALUE MISMATCHES in {len(mismatches)} columns:")
        for col, n in mismatches:
            print(f"  {col}: {n} rows differ")
    else:
        print("\nAll values match!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        if len(sys.argv) != 4:
            print("Usage: python test_staged_join_template.py compare <old_gcs_path> <new_gcs_path>")
            sys.exit(1)
        compare_parquet_outputs(sys.argv[2], sys.argv[3])
    else:
        results = []
        results.append(test_small_case_identical())
        results.append(test_large_case_structure())
        results.append(test_staged_join_references())

        print()
        if all(results):
            print("All tests passed!")
        else:
            print("Some tests FAILED")
            sys.exit(1)
