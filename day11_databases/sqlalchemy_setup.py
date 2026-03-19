# ============================================================
# SQLALCHEMY SETUP
# Day 11: Databases
# Author: Sheshikala
# Topic: SQLAlchemy ORM setup for ML Experiment Tracker
# ============================================================

# SQLAlchemy is a Python ORM (Object Relational Mapper) that
# lets you interact with PostgreSQL using Python classes instead
# of writing raw SQL. I found this much cleaner than writing
# SQL strings manually because you get type checking, IDE
# autocomplete, and automatic query building. This file defines
# the database models for the ML Experiment Tracker schema
# and shows how to create tables, insert records, and run
# queries using SQLAlchemy's declarative base pattern.
# The setup uses SQLite as a fallback so it runs without
# a PostgreSQL server installed.

import os
import json
from datetime import datetime

try:
    from sqlalchemy import (
        create_engine, Column, Integer, String, Float,
        DateTime, ForeignKey, Text, JSON
    )
    from sqlalchemy.orm import declarative_base, sessionmaker, relationship
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    print("sqlalchemy not installed. Run: pip install sqlalchemy")


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_engine(use_sqlite=True):
    """
    Creates and returns a SQLAlchemy engine.
    Uses SQLite by default so the file runs without PostgreSQL.
    Set use_sqlite=False and configure DATABASE_URL environment
    variable to connect to a real PostgreSQL database.

    PostgreSQL URL format:
        postgresql+psycopg2://user:password@localhost:5432/dbname
    """
    if use_sqlite:
        db_path = "ml_tracker.db"
        url     = "sqlite:///" + db_path
        print("Using SQLite database: " + db_path)
    else:
        url = os.environ.get(
            "DATABASE_URL",
            "postgresql+psycopg2://postgres:password@localhost:5432/ml_tracker"
        )
        print("Connecting to PostgreSQL: " + url.split("@")[-1])

    engine = create_engine(url, echo=False)
    return engine


# ============================================================
# ORM MODELS
# ============================================================

if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()

    class Researcher(Base):
        """
        Represents a researcher who runs ML experiments.
        One researcher can belong to many projects and
        run many experiments.
        """
        __tablename__ = "researchers"

        id         = Column(Integer, primary_key=True, autoincrement=True)
        name       = Column(String(100), nullable=False, unique=True)
        department = Column(String(100), nullable=False)
        email      = Column(String(200), nullable=True)
        created_at = Column(DateTime, default=datetime.utcnow)

        experiments = relationship("Experiment", back_populates="researcher")

        def __repr__(self):
            return "Researcher(name=" + self.name + ", dept=" + self.department + ")"

    class Project(Base):
        """
        Represents a research project that groups experiments.
        One project can have many experiments under it.
        """
        __tablename__ = "projects"

        id          = Column(Integer, primary_key=True, autoincrement=True)
        name        = Column(String(200), nullable=False, unique=True)
        domain      = Column(String(100), nullable=False)
        description = Column(Text, nullable=True)
        status      = Column(String(50), default="active")
        created_at  = Column(DateTime, default=datetime.utcnow)

        experiments = relationship("Experiment", back_populates="project")

        def __repr__(self):
            return "Project(name=" + self.name + ", domain=" + self.domain + ")"

    class Dataset(Base):
        """
        Represents a dataset used in ML experiments.
        Stores metadata about size, domain, and split ratios.
        """
        __tablename__ = "datasets"

        id          = Column(Integer, primary_key=True, autoincrement=True)
        name        = Column(String(200), nullable=False, unique=True)
        domain      = Column(String(100), nullable=False)
        size        = Column(String(50), nullable=True)
        description = Column(Text, nullable=True)
        created_at  = Column(DateTime, default=datetime.utcnow)

        experiments = relationship("Experiment", back_populates="dataset")

        def __repr__(self):
            return "Dataset(name=" + self.name + ", domain=" + self.domain + ")"

    class Experiment(Base):
        """
        Represents a single ML experiment run.
        Links a researcher, project, and dataset together
        and stores hyperparameters as JSON for flexibility.
        """
        __tablename__ = "experiments"

        id              = Column(Integer, primary_key=True, autoincrement=True)
        exp_id          = Column(String(20), nullable=False, unique=True)
        model_name      = Column(String(100), nullable=False)
        epochs          = Column(Integer, nullable=False)
        batch_size      = Column(Integer, nullable=False)
        learning_rate   = Column(Float, nullable=False)
        status          = Column(String(50), default="completed")
        hyperparameters = Column(Text, nullable=True)
        notes           = Column(Text, nullable=True)
        created_at      = Column(DateTime, default=datetime.utcnow)

        researcher_id = Column(Integer, ForeignKey("researchers.id"), nullable=False)
        project_id    = Column(Integer, ForeignKey("projects.id"),    nullable=False)
        dataset_id    = Column(Integer, ForeignKey("datasets.id"),    nullable=False)

        researcher = relationship("Researcher", back_populates="experiments")
        project    = relationship("Project",    back_populates="experiments")
        dataset    = relationship("Dataset",    back_populates="experiments")
        metrics    = relationship("Metric",     back_populates="experiment")

        def __repr__(self):
            return ("Experiment(id=" + self.exp_id +
                    ", model=" + self.model_name + ")")

    class Metric(Base):
        """
        Stores evaluation metrics for a completed experiment.
        Each experiment can have multiple metric records
        for different metric types (accuracy, f1, mse etc).
        """
        __tablename__ = "metrics"

        id            = Column(Integer, primary_key=True, autoincrement=True)
        metric_name   = Column(String(50), nullable=False)
        metric_value  = Column(Float, nullable=False)
        split         = Column(String(20), default="test")
        created_at    = Column(DateTime, default=datetime.utcnow)

        experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
        experiment    = relationship("Experiment", back_populates="metrics")

        def __repr__(self):
            return ("Metric(" + self.metric_name +
                    "=" + str(round(self.metric_value, 4)) + ")")


# ============================================================
# DATABASE MANAGER
# ============================================================

class DatabaseManager:
    """
    Manages SQLAlchemy sessions and provides helper methods
    for common database operations on the ML Experiment Tracker.
    Handles session creation, commit, rollback, and close.
    """

    def __init__(self, use_sqlite=True):
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("sqlalchemy is required. Run: pip install sqlalchemy")
        self.engine  = get_engine(use_sqlite)
        Session      = sessionmaker(bind=self.engine)
        self.session = Session()
        Base.metadata.create_all(self.engine)
        print("Database tables created successfully.")

    def add_researcher(self, name, department, email=None):
        """Inserts a new researcher record."""
        researcher = Researcher(name=name, department=department, email=email)
        self.session.add(researcher)
        self.session.commit()
        print("Added researcher: " + name)
        return researcher

    def add_project(self, name, domain, description=None):
        """Inserts a new project record."""
        project = Project(name=name, domain=domain, description=description)
        self.session.add(project)
        self.session.commit()
        print("Added project: " + name)
        return project

    def add_dataset(self, name, domain, size=None, description=None):
        """Inserts a new dataset record."""
        dataset = Dataset(name=name, domain=domain, size=size, description=description)
        self.session.add(dataset)
        self.session.commit()
        print("Added dataset: " + name)
        return dataset

    def add_experiment(self, exp_id, model_name, epochs, batch_size,
                       learning_rate, researcher, project, dataset,
                       notes=None, hyperparameters=None):
        """Inserts a new experiment record linked to researcher, project, dataset."""
        exp = Experiment(
            exp_id        = exp_id,
            model_name    = model_name,
            epochs        = epochs,
            batch_size    = batch_size,
            learning_rate = learning_rate,
            researcher    = researcher,
            project       = project,
            dataset       = dataset,
            notes         = notes,
            hyperparameters = json.dumps(hyperparameters) if hyperparameters else None
        )
        self.session.add(exp)
        self.session.commit()
        print("Added experiment: " + exp_id)
        return exp

    def add_metric(self, experiment, metric_name, metric_value, split="test"):
        """Inserts a metric record linked to an experiment."""
        metric = Metric(
            experiment   = experiment,
            metric_name  = metric_name,
            metric_value = metric_value,
            split        = split
        )
        self.session.add(metric)
        self.session.commit()
        return metric

    def get_all_experiments(self):
        """Returns all experiment records with their metrics."""
        return self.session.query(Experiment).all()

    def get_experiments_by_researcher(self, name):
        """Returns all experiments run by a specific researcher."""
        return (self.session.query(Experiment)
                .join(Researcher)
                .filter(Researcher.name == name)
                .all())

    def get_best_experiment(self, metric_name):
        """Returns the experiment with the highest value for a given metric."""
        result = (self.session.query(Experiment, Metric)
                  .join(Metric)
                  .filter(Metric.metric_name == metric_name)
                  .order_by(Metric.metric_value.desc())
                  .first())
        return result

    def close(self):
        """Closes the database session."""
        self.session.close()
        print("Database session closed.")


# ============================================================
# SAMPLE DATA LOADER
# ============================================================

def load_sample_data(db):
    """
    Inserts sample ML experiment tracker data into the database.
    Creates researchers, projects, datasets, experiments, and metrics.
    """
    print("\n-- Loading sample data --")

    r1 = db.add_researcher("Ananya",  "NLP Research",    "ananya@lab.com")
    r2 = db.add_researcher("Vikram",  "NLP Research",    "vikram@lab.com")
    r3 = db.add_researcher("Priya",   "Computer Vision", "priya@lab.com")
    r4 = db.add_researcher("Rohan",   "ML Engineering",  "rohan@lab.com")

    p1 = db.add_project("NLP-Research",       "NLP",            "Text classification and sentiment analysis")
    p2 = db.add_project("CV-Research",        "Computer Vision","Image recognition using CNNs")
    p3 = db.add_project("TimeSeries-Research","Time Series",    "Forecasting and anomaly detection")

    d1 = db.add_dataset("NLP-Corpus-v2",    "NLP",         "500k", "News articles corpus")
    d2 = db.add_dataset("SentimentData-v1", "NLP",         "100k", "Product review sentiment dataset")
    d3 = db.add_dataset("ImageNet-Subset",  "CV",          "50k",  "ImageNet subset for classification")
    d4 = db.add_dataset("TimeSeriesData-v3","Time Series", "200k", "Hourly time series for forecasting")

    e1 = db.add_experiment("EXP001", "BERT",       10, 32,  2e-5,  r1, p1, d1, notes="Best NLP baseline")
    e2 = db.add_experiment("EXP002", "RoBERTa",    8,  16,  1e-5,  r2, p1, d2, notes="Outperforms BERT on sentiment")
    e3 = db.add_experiment("EXP003", "ResNet50",   30, 64,  0.001, r3, p2, d3, notes="Used data augmentation")
    e4 = db.add_experiment("EXP004", "LSTM",       25, 128, 0.01,  r4, p3, d4, notes="30-step window worked best")

    db.add_metric(e1, "accuracy", 0.91)
    db.add_metric(e1, "f1",       0.89)
    db.add_metric(e2, "accuracy", 0.94)
    db.add_metric(e2, "f1",       0.93)
    db.add_metric(e3, "accuracy", 0.87)
    db.add_metric(e3, "top5_accuracy", 0.96)
    db.add_metric(e4, "mse",      0.023)
    db.add_metric(e4, "mae",      0.14)

    print("Sample data loaded successfully.")


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("SQLALCHEMY SETUP DEMO")
    print("=" * 55)

    db = DatabaseManager(use_sqlite=True)
    load_sample_data(db)

    print("\n-- All Experiments --")
    for exp in db.get_all_experiments():
        print("  " + exp.exp_id + " | " + exp.model_name +
              " | Researcher: " + exp.researcher.name +
              " | Metrics: " + str(len(exp.metrics)))

    print("\n-- Experiments by Ananya --")
    for exp in db.get_experiments_by_researcher("Ananya"):
        print("  " + exp.exp_id + " | " + exp.model_name)

    print("\n-- Best Experiment by Accuracy --")
    result = db.get_best_experiment("accuracy")
    if result:
        exp, metric = result
        print("  " + exp.exp_id + " | " + exp.model_name +
              " | accuracy=" + str(metric.metric_value))

    db.close()

    import os
    if os.path.exists("ml_tracker.db"):
        os.remove("ml_tracker.db")
        print("Cleaned up SQLite file.")

    print("\n-- SQLAlchemy setup demo complete --")


if __name__ == "__main__":
    run_demo()
