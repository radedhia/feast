"""
BigQuery integration test for staged join template change.

Creates synthetic data in BQ session temp tables mimicking the Feast offline flow,
then compares stage counts between the old (single final join) and new (batched)
approaches.

Usage:
    # Small test (fast, verifies correctness)
    python test_staged_join_bq.py --project PROJECT_ID --num-fvs 10 --batch-size 3

    # Production-scale test (slower, verifies stage count reduction)
    python test_staged_join_bq.py --project PROJECT_ID --num-fvs 55 --features-per-fv 100 --entity-rows 500 --row-multiplier 500 --batch-size 15
"""

import argparse
import textwrap

from google.cloud import bigquery


def build_session_queries(
    num_fvs: int,
    num_features_per_fv: int,
    num_entity_rows: int,
    row_multiplier: int = 1,
) -> list[str]:
    """Build the shared setup queries: entity_dataframe + per-FV cleaned tables.

    Args:
        row_multiplier: Multiply the row count in each __cleaned table by this
            factor to simulate larger underlying feature tables. The entity_dataframe
            stays small (it's just the entity keys), but each feature view table
            gets inflated, which increases storage metadata pressure on the planner.
            Use values like 100-1000 to approach production-scale metadata.
    """
    queries = []

    # 1. Entity dataframe with unique IDs per feature view
    unique_id_cols = []
    for i in range(num_fvs):
        unique_id_cols.append(
            f"CONCAT(CAST(patient_id AS STRING), CAST(event_timestamp AS STRING)) "
            f"AS fv_{i}__entity_row_unique_id"
        )

    # The entity_dataframe includes entity_timestamp (an alias of the event
    # timestamp column) just like the real Feast template does.
    queries.append(textwrap.dedent(f"""\
        CREATE TEMP TABLE entity_dataframe AS (
            SELECT
                patient_id,
                event_timestamp,
                event_timestamp AS entity_timestamp,
                {','.join(unique_id_cols)}
            FROM UNNEST(GENERATE_ARRAY(1, {num_entity_rows})) AS patient_id
            CROSS JOIN UNNEST(
                GENERATE_TIMESTAMP_ARRAY('2024-01-01', '2024-12-18', INTERVAL 7 DAY)
            ) AS event_timestamp
        )"""))

    # 2. One __cleaned table per feature view, simulating point-in-time join output.
    #    Each __cleaned table includes entity_timestamp, event_timestamp, and
    #    created_timestamp — matching the real Feast __base CTE output. These
    #    columns overlap with entity_dataframe and will cause "duplicate column"
    #    errors if not properly excluded in the staged join EXCEPT clauses.
    #
    #    When row_multiplier > 1, we inflate each table by cross-joining with a
    #    generated array and then deduplicating back down via ROW_NUMBER(). This
    #    forces BQ to actually materialize larger temp tables (more storage blocks
    #    and metadata) while keeping the final row count correct for the join.
    for i in range(num_fvs):
        feature_cols = ", ".join(
            f"RAND() AS feature_{i}_{j}" for j in range(num_features_per_fv)
        )
        if row_multiplier > 1:
            queries.append(textwrap.dedent(f"""\
                CREATE TEMP TABLE fv_{i}__cleaned AS (
                    WITH inflated AS (
                        SELECT
                            fv_{i}__entity_row_unique_id,
                            patient_id,
                            entity_timestamp,
                            event_timestamp,
                            CURRENT_TIMESTAMP() AS created_timestamp,
                            {feature_cols},
                            ROW_NUMBER() OVER (
                                PARTITION BY fv_{i}__entity_row_unique_id
                            ) AS _rn
                        FROM entity_dataframe
                        CROSS JOIN UNNEST(GENERATE_ARRAY(1, {row_multiplier})) AS _inflator
                    )
                    SELECT * EXCEPT (_rn) FROM inflated WHERE _rn = 1
                )"""))
        else:
            queries.append(textwrap.dedent(f"""\
                CREATE TEMP TABLE fv_{i}__cleaned AS (
                    SELECT
                        fv_{i}__entity_row_unique_id,
                        patient_id,
                        entity_timestamp,
                        event_timestamp,
                        CURRENT_TIMESTAMP() AS created_timestamp,
                        {feature_cols}
                    FROM entity_dataframe
                )"""))

    return queries


def build_old_final_query(num_fvs: int, num_features_per_fv: int) -> str:
    """Build the old single-join final query."""
    all_cols = ["entity_dataframe.patient_id", "entity_dataframe.event_timestamp"]
    for i in range(num_fvs):
        for j in range(num_features_per_fv):
            all_cols.append(f"feature_{i}_{j}")

    joins = []
    for i in range(num_fvs):
        joins.append(textwrap.dedent(f"""\
            LEFT JOIN (
                SELECT * EXCEPT (patient_id)
                FROM fv_{i}__cleaned
            ) USING (fv_{i}__entity_row_unique_id)"""))

    return f"SELECT {', '.join(all_cols)}\nFROM entity_dataframe\n" + "\n".join(joins)


def build_new_final_queries(num_fvs: int, num_features_per_fv: int, batch_size: int) -> list[str]:
    """Build the new staged join queries."""
    queries = []
    batches = [
        list(range(i, min(i + batch_size, num_fvs)))
        for i in range(0, num_fvs, batch_size)
    ]

    # The EXCEPT clause must exclude all columns that exist in both
    # entity_dataframe and __cleaned to avoid duplicate column errors
    # in CREATE TEMP TABLE ... AS (SELECT * ...) statements.
    except_cols = "patient_id, entity_timestamp, event_timestamp, created_timestamp"

    for batch_idx, batch_fvs in enumerate(batches):
        is_first = batch_idx == 0
        is_last = batch_idx == len(batches) - 1

        # Single-batch case uses explicit column names (no duplicate risk),
        # so it only needs to EXCEPT the entity join key like the original.
        # Multi-batch cases use SELECT * so they need the full EXCEPT list.
        cur_except = "patient_id" if (is_first and is_last) else except_cols

        joins = []
        for i in batch_fvs:
            joins.append(textwrap.dedent(f"""\
                LEFT JOIN (
                    SELECT * EXCEPT ({cur_except})
                    FROM fv_{i}__cleaned
                ) USING (fv_{i}__entity_row_unique_id)"""))
        join_sql = "\n".join(joins)

        if is_first and is_last:
            # Only one batch, same as old query
            all_cols = ["entity_dataframe.patient_id", "entity_dataframe.event_timestamp"]
            for i in range(num_fvs):
                for j in range(num_features_per_fv):
                    all_cols.append(f"feature_{i}_{j}")
            queries.append(
                f"SELECT {', '.join(all_cols)}\nFROM entity_dataframe\n{join_sql}"
            )
        elif is_first:
            queries.append(
                f"CREATE TEMP TABLE __staged_join_{batch_idx} AS (\n"
                f"    SELECT *\n    FROM entity_dataframe\n{join_sql}\n)"
            )
        elif is_last:
            all_cols = ["patient_id", "event_timestamp"]
            for i in range(num_fvs):
                for j in range(num_features_per_fv):
                    all_cols.append(f"feature_{i}_{j}")
            queries.append(
                f"SELECT {', '.join(all_cols)}\n"
                f"FROM __staged_join_{batch_idx - 1}\n{join_sql}"
            )
        else:
            queries.append(
                f"CREATE TEMP TABLE __staged_join_{batch_idx} AS (\n"
                f"    SELECT *\n    FROM __staged_join_{batch_idx - 1}\n{join_sql}\n)"
            )

    return queries


def run_in_session(
    client: bigquery.Client, queries: list[str], label: str
) -> dict:
    """Run a list of queries in a BQ session and return stats."""
    # Create session
    session_job = client.query(
        "SELECT 1;",
        job_config=bigquery.QueryJobConfig(create_session=True),
    )
    session_job.result()
    session_id = session_job.session_info.session_id
    print(f"  [{label}] Session: {session_id}")

    job_config = bigquery.QueryJobConfig(
        create_session=False,
        connection_properties=[
            bigquery.query.ConnectionProperty(key="session_id", value=session_id)
        ],
    )

    total_stages = 0
    total_slot_ms = 0
    job_details = []

    result_df = None

    try:
        for i, query in enumerate(queries):
            job = client.query(query, job_config=job_config)
            rows = job.result()

            # Capture the result of the final query (the SELECT)
            is_final = i == len(queries) - 1
            if is_final:
                result_df = rows.to_dataframe()

            # Reload to get full stats
            job = client.get_job(job.job_id)
            num_stages = len(job.query_plan) if job.query_plan else 0
            slot_ms = job.slot_millis or 0

            total_stages += num_stages
            total_slot_ms += slot_ms

            query_preview = query[:80].replace("\n", " ")
            job_details.append({
                "query_num": i + 1,
                "stages": num_stages,
                "slot_ms": slot_ms,
                "preview": query_preview,
            })
            print(f"    Query {i+1}/{len(queries)}: {num_stages} stages, "
                  f"{slot_ms}ms slots — {query_preview}...")
    finally:
        abort_config = bigquery.QueryJobConfig(
            create_session=False,
            connection_properties=[
                bigquery.query.ConnectionProperty(key="session_id", value=session_id)
            ],
        )
        client.query("CALL BQ.ABORT_SESSION();", job_config=abort_config).result()

    return {
        "label": label,
        "total_stages": total_stages,
        "total_slot_ms": total_slot_ms,
        "num_queries": len(queries),
        "details": job_details,
        "result_df": result_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare old vs new Feast final join in BQ")
    parser.add_argument("--project", type=str, default=None, help="GCP project ID")
    parser.add_argument("--batch-size", type=int, default=3, help="Batch size for staged joins")
    parser.add_argument("--num-fvs", type=int, default=10, help="Number of feature views")
    parser.add_argument("--features-per-fv", type=int, default=5, help="Number of features per feature view")
    parser.add_argument("--entity-rows", type=int, default=100, help="Number of entity rows")
    parser.add_argument(
        "--row-multiplier", type=int, default=1,
        help="Multiply rows in each __cleaned table to simulate larger feature tables. "
             "Use 100-1000 to increase storage metadata pressure."
    )
    args = parser.parse_args()

    client = bigquery.Client(project=args.project)
    setup_queries = build_session_queries(
        args.num_fvs, args.features_per_fv, args.entity_rows, args.row_multiplier
    )

    total_cols = args.num_fvs * args.features_per_fv + 2
    print(f"\nConfig: {args.num_fvs} feature views, {args.features_per_fv} features each, "
          f"batch_size={args.batch_size}, entity_rows={args.entity_rows}, "
          f"row_multiplier={args.row_multiplier}")
    print(f"Total columns in output: {total_cols}")
    num_batches = (args.num_fvs + args.batch_size - 1) // args.batch_size
    print(f"Expected batches: {num_batches}")

    # --- Old approach: single final join ---
    print(f"\n{'='*60}")
    print("OLD approach (single final join)")
    print(f"{'='*60}")
    old_final = build_old_final_query(args.num_fvs, args.features_per_fv)
    old_all_queries = setup_queries + [old_final]
    old_stats = run_in_session(client, old_all_queries, "OLD")

    # --- New approach: staged joins ---
    print(f"\n{'='*60}")
    print(f"NEW approach (staged joins, batch_size={args.batch_size})")
    print(f"{'='*60}")
    new_finals = build_new_final_queries(args.num_fvs, args.features_per_fv, args.batch_size)
    new_all_queries = setup_queries + new_finals
    new_stats = run_in_session(client, new_all_queries, "NEW")

    # --- Comparison ---
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    # Stages from setup queries are the same; compare only the final join portion
    old_setup_stages = sum(d["stages"] for d in old_stats["details"][:len(setup_queries)])
    new_setup_stages = sum(d["stages"] for d in new_stats["details"][:len(setup_queries)])
    old_final_stages = old_stats["total_stages"] - old_setup_stages
    new_final_stages = new_stats["total_stages"] - new_setup_stages

    print(f"\n  Setup queries (same for both): {len(setup_queries)} queries")
    print(f"    OLD setup stages: {old_setup_stages}")
    print(f"    NEW setup stages: {new_setup_stages}")
    print(f"\n  Final join queries:")
    print(f"    OLD: 1 query, {old_final_stages} stages")
    print(f"    NEW: {len(new_finals)} queries, {new_final_stages} total stages")
    print(f"\n  Total stages:")
    print(f"    OLD: {old_stats['total_stages']}")
    print(f"    NEW: {new_stats['total_stages']}")
    print(f"    Reduction: {old_stats['total_stages'] - new_stats['total_stages']} "
          f"({100 * (1 - new_stats['total_stages'] / old_stats['total_stages']):.1f}%)")
    print(f"\n  Total slot-ms:")
    print(f"    OLD: {old_stats['total_slot_ms']}")
    print(f"    NEW: {new_stats['total_slot_ms']}")

    # --- Data comparison ---
    print(f"\n{'='*60}")
    print("DATA COMPARISON")
    print(f"{'='*60}")

    old_df = old_stats["result_df"]
    new_df = new_stats["result_df"]

    if old_df is None or new_df is None:
        print("  ERROR: Could not capture result DataFrames")
        return

    # Compare schemas
    old_cols = sorted(old_df.columns.tolist())
    new_cols = sorted(new_df.columns.tolist())
    if old_cols != new_cols:
        missing = set(old_cols) - set(new_cols)
        extra = set(new_cols) - set(old_cols)
        print(f"  SCHEMA MISMATCH")
        if missing:
            print(f"    Missing in new: {missing}")
        if extra:
            print(f"    Extra in new: {extra}")
    else:
        print(f"  Schema: MATCH ({len(old_cols)} columns)")

    # Compare row counts
    if len(old_df) != len(new_df):
        print(f"  Row count: MISMATCH (old={len(old_df)}, new={len(new_df)})")
    else:
        print(f"  Row count: MATCH ({len(old_df)} rows)")

    # Sort both identically and compare values
    sort_cols = ["patient_id", "event_timestamp"]
    common_cols = sorted(set(old_cols) & set(new_cols))
    old_df = old_df[common_cols].sort_values(sort_cols).reset_index(drop=True)
    new_df = new_df[common_cols].sort_values(sort_cols).reset_index(drop=True)

    # Feature columns use RAND(), so we can't compare values directly.
    # Instead verify the non-random columns match and that random columns
    # are present and non-null in both.
    non_random_cols = [c for c in common_cols if c in sort_cols]
    random_cols = [c for c in common_cols if c not in sort_cols]

    non_random_match = old_df[non_random_cols].equals(new_df[non_random_cols])
    print(f"  Entity columns (patient_id, event_timestamp): "
          f"{'MATCH' if non_random_match else 'MISMATCH'}")

    old_nulls = old_df[random_cols].isnull().sum().sum()
    new_nulls = new_df[random_cols].isnull().sum().sum()
    print(f"  Feature columns: {len(random_cols)} columns")
    print(f"    OLD nulls: {old_nulls}")
    print(f"    NEW nulls: {new_nulls}")
    print(f"    Null counts match: {'YES' if old_nulls == new_nulls else 'NO'}")

    if non_random_match and old_nulls == new_nulls and old_cols == new_cols and len(old_df) == len(new_df):
        print(f"\n  RESULT: PASS — outputs are structurally identical")
    else:
        print(f"\n  RESULT: FAIL — outputs differ")


if __name__ == "__main__":
    main()
