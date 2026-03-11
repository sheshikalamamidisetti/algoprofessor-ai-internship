-- ============================================
-- MIGRATION 002: ADD INDEXES
-- Day 11: PostgreSQL Databases
-- Description: Add all indexes for performance
-- Date: 2026-03-11
-- Run after: 001_initial.sql
-- ============================================

-- ============================================
-- STEP 1: RESEARCHERS INDEXES
-- ============================================

CREATE INDEX IF NOT EXISTS idx_researchers_email
ON researchers(email);

CREATE INDEX IF NOT EXISTS idx_researchers_department
ON researchers(department);

-- ============================================
-- STEP 2: ML_PROJECTS INDEXES
-- ============================================

CREATE INDEX IF NOT EXISTS idx_projects_domain
ON ml_projects(domain);

CREATE INDEX IF NOT EXISTS idx_projects_active
ON ml_projects(is_active);

-- ============================================
-- STEP 3: DATASETS INDEXES
-- ============================================

CREATE INDEX IF NOT EXISTS idx_datasets_project
ON datasets(project_id);

CREATE INDEX IF NOT EXISTS idx_datasets_task_type
ON datasets(task_type);

-- GIN index for JSONB metadata
CREATE INDEX IF NOT EXISTS idx_datasets_metadata
ON datasets USING GIN (metadata);

-- ============================================
-- STEP 4: EXPERIMENTS INDEXES
-- ============================================

CREATE INDEX IF NOT EXISTS idx_experiments_project
ON experiments(project_id);

CREATE INDEX IF NOT EXISTS idx_experiments_status
ON experiments(status);

-- Hash index for exact model_type lookups
CREATE INDEX IF NOT EXISTS idx_experiments_model_hash
ON experiments USING HASH (model_type);

-- GIN index for JSONB hyperparameters
CREATE INDEX IF NOT EXISTS idx_experiments_hparams
ON experiments USING GIN (hyperparameters);

-- Partial index — only completed experiments
CREATE INDEX IF NOT EXISTS idx_experiments_completed_partial
ON experiments(created_at DESC)
WHERE status = 'completed';

-- Composite index — project + status together
CREATE INDEX IF NOT EXISTS idx_experiments_project_status
ON experiments(project_id, status);

-- ============================================
-- STEP 5: MODEL_METRICS INDEXES
-- ============================================

CREATE INDEX IF NOT EXISTS idx_metrics_experiment
ON model_metrics(experiment_id);

-- B-Tree on f1_score for leaderboard queries
CREATE INDEX IF NOT EXISTS idx_metrics_f1_btree
ON model_metrics(f1_score DESC);

-- Partial index — test split only (most queried)
CREATE INDEX IF NOT EXISTS idx_metrics_test_split_partial
ON model_metrics(f1_score DESC)
WHERE split = 'test';

-- Composite — experiment + split
CREATE INDEX IF NOT EXISTS idx_metrics_exp_split
ON model_metrics(experiment_id, split);

-- BRIN index for recorded_at (large table, sequential)
CREATE INDEX IF NOT EXISTS idx_metrics_recorded_brin
ON model_metrics USING BRIN (recorded_at);

-- ============================================
-- STEP 6: VECTOR_EMBEDDINGS INDEXES
-- ============================================

CREATE INDEX IF NOT EXISTS idx_embeddings_dataset
ON vector_embeddings(dataset_id);

CREATE INDEX IF NOT EXISTS idx_embeddings_model
ON vector_embeddings(model_name);

-- GiST index for full-text search on chunk_text
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_gist
ON vector_embeddings USING GiST (to_tsvector('english', chunk_text));

-- ============================================
-- VERIFY ALL INDEXES
-- ============================================
SELECT
    indexname  AS index_name,
    tablename  AS on_table
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;

SELECT 'Migration 002 — All indexes created successfully!' AS status;
SELECT 'Total index types: B-Tree, Hash, GIN, GiST, BRIN, Partial, Composite' AS summary;
