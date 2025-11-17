"""Database session management and initialization."""

from typing import Generator, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base


class DatabaseConfig:
    """Database configuration settings."""

    def __init__(self, database_url: Optional[str] = None):
        """Initialize database configuration.

        Args:
            database_url: Database connection URL. Defaults to SQLite in-memory database.
        """
        self.database_url = database_url or "sqlite:///./pv_circularity.db"
        self.connect_args = {}

        # Use in-memory SQLite for testing
        if self.database_url.startswith("sqlite"):
            self.connect_args = {"check_same_thread": False}


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """Initialize database manager.

        Args:
            config: Database configuration. If None, uses default config.
        """
        self.config = config or DatabaseConfig()
        self.engine = create_engine(
            self.config.database_url,
            connect_args=self.config.connect_args,
            echo=False,  # Set to True for SQL query logging
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self) -> None:
        """Drop all database tables. Use with caution!"""
        Base.metadata.drop_all(bind=self.engine)

    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session.

        Yields:
            Database session.
        """
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def init_db(database_url: Optional[str] = None, create_tables: bool = True) -> DatabaseManager:
    """Initialize the database.

    Args:
        database_url: Database connection URL.
        create_tables: Whether to create tables immediately.

    Returns:
        Initialized DatabaseManager instance.
    """
    global _db_manager
    config = DatabaseConfig(database_url)
    _db_manager = DatabaseManager(config)

    if create_tables:
        _db_manager.create_tables()

    return _db_manager


def get_db_session() -> Generator[Session, None, None]:
    """Get a database session using the global database manager.

    Yields:
        Database session.

    Raises:
        RuntimeError: If database has not been initialized.
    """
    if _db_manager is None:
        raise RuntimeError(
            "Database not initialized. Call init_db() first or create a DatabaseManager instance."
        )
    yield from _db_manager.get_session()


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance.

    Returns:
        DatabaseManager instance.

    Raises:
        RuntimeError: If database has not been initialized.
    """
    if _db_manager is None:
        raise RuntimeError(
            "Database not initialized. Call init_db() first or create a DatabaseManager instance."
        )
    return _db_manager
