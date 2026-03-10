-- ============================================
-- ALL PARTITIONING TYPES — ML / DS CONTEXT
-- Day 11: PostgreSQL + Async SQLAlchemy
-- ============================================

-- ============================================
-- PART 1: RANGE PARTITIONING (BY DATE)
-- ============================================
-- Use case: experiment_logs grow over time
-- Partition by month for fast date-range queries

CREATE TABLE experiment_logs (
    log_id          SERIAL,
    experiment_id   INT,
    log_level       VARCHAR(10),     -- INFO, WARNING, ERROR
    message         TEXT,
    logged_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
PARTITION BY RANGE (logged_at);

-- Monthly partitions
CREATE TABLE experiment_logs_2026_jan PARTITION OF experiment_logs
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

CREATE TABLE experiment_logs_2026_feb PARTITION OF experiment_logs
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');

CREATE TABLE experiment_logs_2026_mar PARTITION OF experiment_logs
FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');

CREATE TABLE experiment_logs_future PARTITION OF experiment_logs
FOR VALUES FROM ('2026-04-01') TO ('2030-01-01');

-- Insert sample logs
INSERT INTO experiment_logs (experiment_id, log_level, message, logged_at) VALUES
(1, 'INFO',    'Training started: BERT-base, lr=2e-5',              '2026-01-15 09:00:00'),
(1, 'INFO',    'Epoch 1/3 complete — train_loss=0.45',              '2026-01-15 09:45:00'),
(2, 'INFO',    'Training started: RoBERTa, lr=1e-5',               '2026-02-10 10:00:00'),
(2, 'WARNING', 'GPU memory at 92% — consider reducing batch size',  '2026-02-10 11:00:00'),
(3, 'INFO',    'Training started: DistilBERT',                      '2026-03-03 09:30:00'),
(3, 'ERROR',   'Validation loss diverged at epoch 3',               '2026-03-03 10:30:00'),
(4, 'INFO',    'Training complete: ResNet50, best_f1=0.858',        '2026-03-04 18:00:00');

-- Query — PostgreSQL automatically scans only March partition
SELECT * FROM experiment_logs
WHERE logged_at >= '2026-03-01' AND logged_at < '2026-04-01';

-- ============================================
-- PART 2: LIST PARTITIONING (BY CATEGORY)
-- ============================================
-- Use case: model_registry partitioned by task type
-- Each task type in a separate physical partition

CREATE TABLE model_registry (
    model_id        SERIAL,
    model_name      VARCHAR(150),
    task_type       VARCHAR(50),     -- NLP, CV, Tabular, RL
    framework       VARCHAR(50),
    accuracy        DECIMAL(6,4),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
PARTITION BY LIST (task_type);

CREATE TABLE model_registry_nlp
PARTITION OF model_registry
FOR VALUES IN ('NLP');

CREATE TABLE model_registry_cv
PARTITION OF model_registry
FOR VALUES IN ('CV');

CREATE TABLE model_registry_tabular
PARTITION OF model_registry
FOR VALUES IN ('Tabular');

CREATE TABLE model_registry_other
PARTITION OF model_registry
DEFAULT;

-- Insert models
INSERT INTO model_registry (model_name, task_type, framework, accuracy) VALUES
('BERT-base Sentiment',         'NLP',     'PyTorch',     0.9020),
('RoBERTa Sentiment',           'NLP',     'PyTorch',     0.9360),
('DistilBERT Fast',             'NLP',     'PyTorch',     0.8985),
('ResNet50 X-Ray',              'CV',      'PyTorch',     0.8580),
('DenseNet121 Medical',         'CV',      'TensorFlow',  0.8820),
('XGBoost Sales',               'Tabular', 'XGBoost',     0.9130),
('LSTM Forecast',               'Tabular', 'Keras',       0.9015),
('LightGBM Fraud',              'Tabular', 'LightGBM',    0.9279),
('RAG-Chroma Pipeline',         'NLP',     'LangChain',   0.8680),
('DQN Agent v1',                'RL',      'Stable-Baselines3', 0.7200);

-- Query — hits only the NLP partition
SELECT model_name, framework, accuracy
FROM model_registry
WHERE task_type = 'NLP'
ORDER BY accuracy DESC;

-- ============================================
-- PART 3: HASH PARTITIONING (BY ID)
-- ============================================
-- Use case: distribute raw_predictions evenly
-- Good when no natural range or list key

CREATE TABLE raw_predictions (
    prediction_id   INT,
    experiment_id   INT,
    sample_index    INT,
    predicted_label INT,
    confidence      DECIMAL(6,4),
    true_label      INT
)
PARTITION BY HASH (experiment_id);

-- 4 hash partitions for even distribution
CREATE TABLE raw_predictions_p0 PARTITION OF raw_predictions
FOR VALUES WITH (MODULUS 4, REMAINDER 0);

CREATE TABLE raw_predictions_p1 PARTITION OF raw_predictions
FOR VALUES WITH (MODULUS 4, REMAINDER 1);

CREATE TABLE raw_predictions_p2 PARTITION OF raw_predictions
FOR VALUES WITH (MODULUS 4, REMAINDER 2);

CREATE TABLE raw_predictions_p3 PARTITION OF raw_predictions
FOR VALUES WITH (MODULUS 4, REMAINDER 3);

-- Insert sample predictions
INSERT INTO raw_predictions (prediction_id, experiment_id, sample_index, predicted_label, confidence, true_label) VALUES
(1, 1, 100, 1, 0.9821, 1),
(2, 1, 101, 0, 0.7430, 0),
(3, 2, 100, 1, 0.9910, 1),
(4, 2, 101, 1, 0.8120, 0),  -- wrong prediction
(5, 3, 100, 0, 0.9540, 0),
(6, 4, 100, 1, 0.8870, 1),
(7, 5, 100, 0, 0.9960, 0),
(8, 6, 100, 1, 0.9120, 1);

-- Query — PostgreSQL routes to correct partition
SELECT *
FROM raw_predictions
WHERE experiment_id = 2;

-- ============================================
-- PART 4: SUB-PARTITIONING (RANGE → LIST)
-- ============================================
-- Use case: large metric_archive partitioned by
-- year (RANGE) then by split (LIST)

CREATE TABLE metric_archive (
    archive_id      SERIAL,
    experiment_id   INT,
    recorded_at     DATE,
    split           VARCHAR(20),
    f1_score        DECIMAL(6,4),
    accuracy        DECIMAL(6,4)
)
PARTITION BY RANGE (recorded_at);

-- Year-level partition
CREATE TABLE metric_archive_2026
PARTITION OF metric_archive
FOR VALUES FROM ('2026-01-01') TO ('2027-01-01')
PARTITION BY LIST (split);

-- Sub-partitions by split type
CREATE TABLE metric_archive_2026_train
PARTITION OF metric_archive_2026
FOR VALUES IN ('train');

CREATE TABLE metric_archive_2026_val
PARTITION OF metric_archive_2026
FOR VALUES IN ('validation');

CREATE TABLE metric_archive_2026_test
PARTITION OF metric_archive_2026
FOR VALUES IN ('test');

-- Insert archived metrics
INSERT INTO metric_archive (experiment_id, recorded_at, split, f1_score, accuracy) VALUES
(1, '2026-01-15', 'train',      0.9395, 0.9420),
(1, '2026-01-15', 'validation', 0.9065, 0.9110),
(1, '2026-01-15', 'test',       0.9020, 0.9050),
(2, '2026-02-10', 'train',      0.9660, 0.9680),
(2, '2026-02-10', 'validation', 0.9410, 0.9430),
(2, '2026-02-10', 'test',       0.9360, 0.9380);

-- Query hits exact sub-partition
SELECT * FROM metric_archive
WHERE recorded_at >= '2026-01-01'
  AND split = 'test';

-- ============================================
-- PART 5: VIEW ALL PARTITIONS
-- ============================================
SELECT
    parent.relname  AS parent_table,
    child.relname   AS partition_name
FROM pg_inherits
JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
JOIN pg_class child  ON pg_inherits.inhrelid  = child.oid
ORDER BY parent_table, partition_name;

-- ============================================
-- PART 6: QUERY PERFORMANCE — PARTITION PRUNING
-- ============================================
-- PostgreSQL skips irrelevant partitions automatically

EXPLAIN ANALYZE
SELECT * FROM experiment_logs
WHERE logged_at >= '2026-03-01';

EXPLAIN ANALYZE
SELECT * FROM model_registry
WHERE task_type = 'NLP';

EXPLAIN ANALYZE
SELECT * FROM raw_predictions
WHERE experiment_id = 3;

-- ============================================
-- SUMMARY
-- ============================================
SELECT 'All partition types created successfully!' AS message;
SELECT 'Types: RANGE (date logs), LIST (task_type), HASH (predictions), SUB-PARTITION (archive)' AS summary;
