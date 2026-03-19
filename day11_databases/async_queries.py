# ============================================================
# ASYNC QUERIES
# Day 11: Databases
# Author: Sheshikala
# Topic: Async SQLAlchemy queries for ML Experiment Tracker
# ============================================================

# Async SQLAlchemy allows database queries to run without
# blocking the rest of the program. This is important when
# building APIs or dashboards that need to handle multiple
# requests at the same time. Instead of waiting for one
# query to finish before starting the next, async lets
# multiple queries run concurrently. This file demonstrates
# async SQLAlchemy with asyncio using the same ML Experiment
# Tracker schema. It also shows synchronous fallback patterns
# for environments where async is not needed. The async
# version uses aiosqlite for SQLite and asyncpg for PostgreSQL.

import asyncio
import time
import os

try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import select, func, text
    from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
    from sqlalchemy.orm import declarative_base, relationship
    from datetime import datetime
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False


# ============================================================
# ASYNC ENGINE SETUP
# ============================================================

def get_async_engine(use_sqlite=True):
    """
    Creates an async SQLAlchemy engine.
    Uses aiosqlite for SQLite (no server needed) or
    asyncpg for PostgreSQL (requires server running).

    SQLite async URL format : sqlite+aiosqlite:///filename.db
    PostgreSQL async URL format: postgresql+asyncpg://user:pass@host/db
    """
    if use_sqlite:
        if not AIOSQLITE_AVAILABLE:
            print("aiosqlite not installed. Run: pip install aiosqlite")
            return None
        url = "sqlite+aiosqlite:///ml_tracker_async.db"
        print("Async engine using SQLite: ml_tracker_async.db")
    else:
        url = os.environ.get(
            "ASYNC_DATABASE_URL",
            "postgresql+asyncpg://postgres:password@localhost:5432/ml_tracker"
        )
        print("Async engine using PostgreSQL: " + url.split("@")[-1])

    from sqlalchemy.ext.asyncio import create_async_engine
    engine = create_async_engine(url, echo=False)
    return engine


# ============================================================
# ASYNC MODELS (same schema as sqlalchemy_setup.py)
# ============================================================

if ASYNC_AVAILABLE:
    AsyncBase = declarative_base()

    class AsyncResearcher(AsyncBase):
        __tablename__ = "researchers"
        id         = Column(Integer, primary_key=True, autoincrement=True)
        name       = Column(String(100), nullable=False, unique=True)
        department = Column(String(100), nullable=False)
        created_at = Column(DateTime, default=datetime.utcnow)
        experiments = relationship("AsyncExperiment", back_populates="researcher",
                                   lazy="selectin")

    class AsyncProject(AsyncBase):
        __tablename__ = "projects"
        id     = Column(Integer, primary_key=True, autoincrement=True)
        name   = Column(String(200), nullable=False, unique=True)
        domain = Column(String(100), nullable=False)
        experiments = relationship("AsyncExperiment", back_populates="project",
                                   lazy="selectin")

    class AsyncDataset(AsyncBase):
        __tablename__ = "datasets"
        id     = Column(Integer, primary_key=True, autoincrement=True)
        name   = Column(String(200), nullable=False, unique=True)
        domain = Column(String(100), nullable=False)
        experiments = relationship("AsyncExperiment", back_populates="dataset",
                                   lazy="selectin")

    class AsyncExperiment(AsyncBase):
        __tablename__ = "experiments"
        id            = Column(Integer, primary_key=True, autoincrement=True)
        exp_id        = Column(String(20), nullable=False, unique=True)
        model_name    = Column(String(100), nullable=False)
        epochs        = Column(Integer, nullable=False)
        batch_size    = Column(Integer, nullable=False)
        learning_rate = Column(Float, nullable=False)
        created_at    = Column(DateTime, default=datetime.utcnow)

        researcher_id = Column(Integer, ForeignKey("researchers.id"), nullable=False)
        project_id    = Column(Integer, ForeignKey("projects.id"),    nullable=False)
        dataset_id    = Column(Integer, ForeignKey("datasets.id"),    nullable=False)

        researcher = relationship("AsyncResearcher", back_populates="experiments",
                                  lazy="selectin")
        project    = relationship("AsyncProject",    back_populates="experiments",
                                  lazy="selectin")
        dataset    = relationship("AsyncDataset",    back_populates="experiments",
                                  lazy="selectin")
        metrics    = relationship("AsyncMetric",     back_populates="experiment",
                                  lazy="selectin")

    class AsyncMetric(AsyncBase):
        __tablename__ = "metrics"
        id            = Column(Integer, primary_key=True, autoincrement=True)
        metric_name   = Column(String(50), nullable=False)
        metric_value  = Column(Float, nullable=False)
        experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
        experiment    = relationship("AsyncExperiment", back_populates="metrics",
                                     lazy="selectin")


# ============================================================
# ASYNC DATABASE OPERATIONS
# ============================================================

async def create_tables(engine):
    """Creates all tables in the async database."""
    async with engine.begin() as conn:
        await conn.run_sync(AsyncBase.metadata.create_all)
    print("Async tables created.")


async def insert_sample_data(engine):
    """
    Inserts sample researchers, projects, datasets,
    experiments, and metrics using async session.
    """
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        r1 = AsyncResearcher(name="Ananya", department="NLP Research")
        r2 = AsyncResearcher(name="Vikram", department="NLP Research")
        r3 = AsyncResearcher(name="Priya",  department="Computer Vision")
        session.add_all([r1, r2, r3])
        await session.flush()

        p1 = AsyncProject(name="NLP-Research", domain="NLP")
        p2 = AsyncProject(name="CV-Research",  domain="Computer Vision")
        session.add_all([p1, p2])
        await session.flush()

        d1 = AsyncDataset(name="NLP-Corpus-v2",   domain="NLP")
        d2 = AsyncDataset(name="ImageNet-Subset",  domain="CV")
        session.add_all([d1, d2])
        await session.flush()

        e1 = AsyncExperiment(
            exp_id="EXP001", model_name="BERT", epochs=10,
            batch_size=32, learning_rate=2e-5,
            researcher_id=r1.id, project_id=p1.id, dataset_id=d1.id
        )
        e2 = AsyncExperiment(
            exp_id="EXP002", model_name="RoBERTa", epochs=8,
            batch_size=16, learning_rate=1e-5,
            researcher_id=r2.id, project_id=p1.id, dataset_id=d1.id
        )
        e3 = AsyncExperiment(
            exp_id="EXP003", model_name="ResNet50", epochs=30,
            batch_size=64, learning_rate=0.001,
            researcher_id=r3.id, project_id=p2.id, dataset_id=d2.id
        )
        session.add_all([e1, e2, e3])
        await session.flush()

        metrics_data = [
            AsyncMetric(experiment_id=e1.id, metric_name="accuracy", metric_value=0.91),
            AsyncMetric(experiment_id=e1.id, metric_name="f1",       metric_value=0.89),
            AsyncMetric(experiment_id=e2.id, metric_name="accuracy", metric_value=0.94),
            AsyncMetric(experiment_id=e2.id, metric_name="f1",       metric_value=0.93),
            AsyncMetric(experiment_id=e3.id, metric_name="accuracy", metric_value=0.87),
        ]
        session.add_all(metrics_data)
        await session.commit()
        print("Sample data inserted asynchronously.")


async def query_all_experiments(engine):
    """
    Queries all experiments and prints their details.
    Demonstrates basic async SELECT with relationship loading.
    """
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        result = await session.execute(select(AsyncExperiment))
        experiments = result.scalars().all()

        print("\n-- All Experiments (async query) --")
        for exp in experiments:
            metric_count = len(exp.metrics)
            print("  " + exp.exp_id +
                  " | " + exp.model_name +
                  " | " + exp.researcher.name +
                  " | metrics=" + str(metric_count))
        return experiments


async def query_experiments_by_researcher(engine, researcher_name):
    """
    Queries experiments filtered by researcher name.
    Demonstrates async JOIN query with filter condition.
    """
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        result = await session.execute(
            select(AsyncExperiment)
            .join(AsyncResearcher)
            .where(AsyncResearcher.name == researcher_name)
        )
        experiments = result.scalars().all()
        print("\n-- Experiments by " + researcher_name + " (async) --")
        for exp in experiments:
            print("  " + exp.exp_id + " | " + exp.model_name)
        return experiments


async def query_best_accuracy(engine):
    """
    Finds the experiment with the highest accuracy metric.
    Demonstrates async aggregation with ORDER BY and LIMIT.
    """
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        result = await session.execute(
            select(AsyncExperiment, AsyncMetric)
            .join(AsyncMetric)
            .where(AsyncMetric.metric_name == "accuracy")
            .order_by(AsyncMetric.metric_value.desc())
            .limit(1)
        )
        row = result.first()
        if row:
            exp, metric = row
            print("\n-- Best Accuracy (async) --")
            print("  " + exp.exp_id + " | " + exp.model_name +
                  " | accuracy=" + str(metric.metric_value))
        return row


async def run_concurrent_queries(engine):
    """
    Runs multiple queries concurrently using asyncio.gather.
    This is the key benefit of async: all three queries
    execute at the same time instead of one after another,
    which significantly reduces total wait time on a real
    database with network latency.
    """
    print("\n-- Running 3 queries concurrently --")
    start = time.time()

    results = await asyncio.gather(
        query_all_experiments(engine),
        query_experiments_by_researcher(engine, "Ananya"),
        query_best_accuracy(engine)
    )

    elapsed = round(time.time() - start, 3)
    print("\nAll 3 queries completed in " + str(elapsed) + "s (concurrent)")
    return results


# ============================================================
# MOCK ASYNC DEMO (works without aiosqlite)
# ============================================================

async def mock_async_demo():
    """
    Simulates async database operations without a real database.
    Shows the async/await pattern and timing benefits.
    """
    print("\n-- Mock Async Demo (no database required) --")

    async def mock_query(query_name, delay=0.05):
        await asyncio.sleep(delay)
        return query_name + " completed"

    print("Running queries sequentially:")
    start = time.time()
    r1 = await mock_query("SELECT all experiments",   0.1)
    r2 = await mock_query("SELECT by researcher",     0.1)
    r3 = await mock_query("SELECT best accuracy",     0.1)
    sequential_time = round(time.time() - start, 3)
    print("  " + r1)
    print("  " + r2)
    print("  " + r3)
    print("  Sequential time: " + str(sequential_time) + "s")

    print("\nRunning queries concurrently:")
    start = time.time()
    results = await asyncio.gather(
        mock_query("SELECT all experiments",   0.1),
        mock_query("SELECT by researcher",     0.1),
        mock_query("SELECT best accuracy",     0.1)
    )
    concurrent_time = round(time.time() - start, 3)
    for r in results:
        print("  " + r)
    print("  Concurrent time: " + str(concurrent_time) + "s")
    print("  Speedup factor : " + str(round(sequential_time / concurrent_time, 1)) + "x")


# ============================================================
# MAIN ASYNC RUNNER
# ============================================================

async def run_async_demo():
    """Main async demo function."""
    print("=" * 55)
    print("ASYNC SQLALCHEMY QUERIES DEMO")
    print("=" * 55)

    await mock_async_demo()

    if not ASYNC_AVAILABLE or not AIOSQLITE_AVAILABLE:
        print("\nSkipping real async DB demo.")
        print("Install: pip install sqlalchemy[asyncio] aiosqlite")
        return

    engine = get_async_engine(use_sqlite=True)
    if engine is None:
        return

    await create_tables(engine)
    await insert_sample_data(engine)
    await run_concurrent_queries(engine)

    await engine.dispose()

    if os.path.exists("ml_tracker_async.db"):
        os.remove("ml_tracker_async.db")
        print("\nCleaned up async SQLite file.")

    print("\n-- Async queries demo complete --")


def run_demo():
    asyncio.run(run_async_demo())


if __name__ == "__main__":
    run_demo()
