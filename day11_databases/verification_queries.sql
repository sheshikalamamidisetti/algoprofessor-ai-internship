-- ============================================
-- VERIFICATION QUERIES
-- Day 11: PostgreSQL + Async SQLAlchemy
-- Run this AFTER: db_schema.sql → sample_data.sql
-- ============================================

-- ============================================
-- STEP 1: VERIFY TABLES + ROW COUNTS
-- ============================================
SELECT 'researchers'       AS table_name, COUNT(*) AS records FROM researchers
UNION ALL
SELECT 'ml_projects',      COUNT(*) FROM ml_projects
UNION ALL
SELECT 'datasets',         COUNT(*) FROM datasets
UNION ALL
SELECT 'experiments',      COUNT(*) FROM experiments
UNION ALL
SELECT 'model_metrics',    COUNT(*) FROM model_metrics
UNION ALL
SELECT 'vector_embeddings',COUNT(*) FROM vector_embeddings
ORDER BY table_name;

-- ============================================
-- STEP 2: VERIFY VIEWS EXIST + RETURN DATA
-- ============================================

-- View 1: Experiment leaderboard
SELECT 'experiment_leaderboard' AS view_name;
SELECT project_name, experiment_name, model_type, f1_score
FROM experiment_leaderboard
LIMIT 5;

-- View 2: Dataset summary
SELECT 'dataset_summary' AS view_name;
SELECT name, task_type, total_experiments, best_f1
FROM dataset_summary
ORDER BY best_f1 DESC NULLS LAST;

-- View 3: Researcher activity
SELECT 'researcher_activity' AS view_name;
SELECT username, department, total_experiments, avg_f1_score
FROM researcher_activity
ORDER BY avg_f1_score DESC NULLS LAST;

-- ============================================
-- STEP 3: VERIFY JSONB COLUMNS
-- ============================================

-- Hyperparameters readable?
SELECT
    experiment_name,
    hyperparameters->>'lr'         AS lr,
    hyperparameters->>'epochs'     AS epochs,
    hyperparameters->>'batch_size' AS batch_size
FROM experiments
WHERE hyperparameters IS NOT NULL
LIMIT 5;

-- Dataset metadata readable?
SELECT
    name,
    metadata->>'license' AS license,
    metadata->>'url'     AS url
FROM datasets
WHERE metadata IS NOT NULL;

-- ============================================
-- STEP 4: VERIFY RELATIONSHIPS (FOREIGN KEYS)
-- ============================================

-- Every experiment has a valid project
SELECT
    e.experiment_id,
    e.experiment_name,
    p.project_name
FROM experiments e
JOIN ml_projects p ON e.project_id = p.project_id
LIMIT 5;

-- Every metric has a valid experiment
SELECT
    mm.metric_id,
    mm.split,
    mm.f1_score,
    e.experiment_name
FROM model_metrics mm
JOIN experiments e ON mm.experiment_id = e.experiment_id
LIMIT 5;

-- ============================================
-- STEP 5: VERIFY FUNCTIONS
-- ============================================

-- Test fn_best_experiment for project 1
SELECT 'fn_best_experiment(1)' AS function_test;
SELECT * FROM fn_best_experiment(1);

-- Test fn_experiment_duration
SELECT 'fn_experiment_duration' AS function_test;
SELECT
    experiment_id,
    experiment_name,
    fn_experiment_duration(experiment_id) AS duration_minutes
FROM experiments
WHERE status = 'completed'
LIMIT 5;

-- ============================================
-- STEP 6: VERIFY INDEXES
-- ============================================
SELECT
    indexname   AS index_name,
    tablename   AS on_table
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;

-- ============================================
-- STEP 7: QUICK ANALYTICS SANITY CHECK
-- ============================================

-- Best model per project
SELECT
    p.project_name,
    e.model_type,
    MAX(mm.f1_score) AS best_f1
FROM experiments e
JOIN ml_projects   p  ON e.project_id    = p.project_id
JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
WHERE mm.split = 'test'
GROUP BY p.project_name, e.model_type
ORDER BY best_f1 DESC;

-- Train vs Test gap (overfitting check)
SELECT
    e.experiment_name,
    MAX(CASE WHEN mm.split='train' THEN mm.f1_score END) AS train_f1,
    MAX(CASE WHEN mm.split='test'  THEN mm.f1_score END) AS test_f1,
    MAX(CASE WHEN mm.split='train' THEN mm.f1_score END)
    - MAX(CASE WHEN mm.split='test' THEN mm.f1_score END) AS overfit_gap
FROM experiments e
JOIN model_metrics mm ON e.experiment_id = mm.experiment_id
GROUP BY e.experiment_id, e.experiment_name
HAVING MAX(CASE WHEN mm.split='train' THEN mm.f1_score END) IS NOT NULL
ORDER BY overfit_gap DESC;

-- ============================================
-- STEP 8: OVERALL HEALTH CHECK
-- ============================================
DO $$
DECLARE
    v_tables    INT;
    v_views     INT;
    v_indexes   INT;
    v_functions INT;
    v_triggers  INT;
BEGIN
    SELECT COUNT(*) INTO v_tables
    FROM information_schema.tables
    WHERE table_schema='public' AND table_type='BASE TABLE';

    SELECT COUNT(*) INTO v_views
    FROM information_schema.views
    WHERE table_schema='public';

    SELECT COUNT(*) INTO v_indexes
    FROM pg_indexes
    WHERE schemaname='public';

    SELECT COUNT(*) INTO v_functions
    FROM information_schema.routines
    WHERE routine_schema='public' AND routine_type='FUNCTION';

    SELECT COUNT(*) INTO v_triggers
    FROM information_schema.triggers
    WHERE trigger_schema='public';

    RAISE NOTICE '========================================';
    RAISE NOTICE 'Day 11 Database — Health Check';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Tables    : %', v_tables;
    RAISE NOTICE 'Views     : %', v_views;
    RAISE NOTICE 'Indexes   : %', v_indexes;
    RAISE NOTICE 'Functions : %', v_functions;
    RAISE NOTICE 'Triggers  : %', v_triggers;
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Status    : ALL SYSTEMS OPERATIONAL';
    RAISE NOTICE '========================================';
END $$;
