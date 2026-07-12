"""
DriveWise Query Logger
───────────────────────
SQLite-backed logging of every query, response, evaluation scores,
and diagnostic metadata. Powers the Analytics dashboard.
"""

import sqlite3, os, json
from datetime import datetime
from config import LOG_DB_PATH


class QueryLogger:

    def __init__(self):
        os.makedirs(os.path.dirname(LOG_DB_PATH), exist_ok=True)
        self._init_db()

    # ──────────────────────────────────────────────────────────────────────────
    # Schema
    # ──────────────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(LOG_DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp            TEXT    NOT NULL,
                    brand                TEXT,
                    model                TEXT,
                    query                TEXT,
                    answer               TEXT,
                    sources_json         TEXT,
                    response_time        REAL,
                    retrieved_count      INTEGER,
                    reranked_count       INTEGER,
                    faithfulness         REAL,
                    context_relevance    REAL,
                    answer_completeness  REAL,
                    overall_score        REAL,
                    failed               INTEGER DEFAULT 0,
                    error_message        TEXT
                )
            """)
            conn.commit()

    # ──────────────────────────────────────────────────────────────────────────
    # Write
    # ──────────────────────────────────────────────────────────────────────────

    def log(self, result: dict, eval_scores: dict | None = None,
            error: Exception | None = None) -> None:
        ev = eval_scores or {}
        with sqlite3.connect(LOG_DB_PATH) as conn:
            conn.execute("""
                INSERT INTO query_logs
                (timestamp, brand, model, query, answer, sources_json,
                 response_time, retrieved_count, reranked_count,
                 faithfulness, context_relevance, answer_completeness,
                 overall_score, failed, error_message)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                datetime.now().isoformat(),
                result.get("brand",           ""),
                result.get("model",           ""),
                result.get("query",           ""),
                result.get("answer",          ""),
                json.dumps(result.get("sources", [])),
                result.get("response_time",    0),
                result.get("retrieved_count",  0),
                result.get("reranked_count",   0),
                ev.get("faithfulness"),
                ev.get("context_relevance"),
                ev.get("answer_completeness"),
                ev.get("overall"),
                1 if error else 0,
                str(error) if error else None,
            ))
            conn.commit()

    # ──────────────────────────────────────────────────────────────────────────
    # Read
    # ──────────────────────────────────────────────────────────────────────────

    def get_logs(self, limit: int = 100) -> list[dict]:
        with sqlite3.connect(LOG_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM query_logs ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        with sqlite3.connect(LOG_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn

            total    = c.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0]
            failed   = c.execute("SELECT COUNT(*) FROM query_logs WHERE failed=1").fetchone()[0]
            avg_time = c.execute(
                "SELECT AVG(response_time) FROM query_logs WHERE failed=0"
            ).fetchone()[0]
            avg_score = c.execute(
                "SELECT AVG(overall_score) FROM query_logs WHERE failed=0 AND overall_score IS NOT NULL"
            ).fetchone()[0]
            top_models = [
                dict(r) for r in c.execute("""
                    SELECT brand, model, COUNT(*) AS count
                    FROM query_logs
                    GROUP BY brand, model
                    ORDER BY count DESC
                    LIMIT 5
                """).fetchall()
            ]
            section_dist = [
                dict(r) for r in c.execute("""
                    SELECT json_extract(value, '$.section') AS section,
                           COUNT(*) AS count
                    FROM query_logs, json_each(sources_json)
                    GROUP BY section
                    ORDER BY count DESC
                    LIMIT 10
                """).fetchall()
            ] if total > 0 else []

        return {
            "total_queries":    total,
            "failed_queries":   failed,
            "success_rate":     round((total - failed) / max(total, 1), 3),
            "avg_response_time": round(avg_time or 0, 2),
            "avg_overall_score": round(avg_score or 0, 3),
            "top_models":        top_models,
            "section_dist":      section_dist,
        }
