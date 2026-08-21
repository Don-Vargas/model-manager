from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import (
    ArtifactModel,
    Base,
    ComponentModel,
    ModelUnitModel,
    RepositoryModel,
)
from app.services.repository_service import RepositoryService

DATA_DIR = Path("/app/data/huggingface")


# ==============================================================================
# FIXTURES DE BD EN MEMORIA (SQLite)
# ==============================================================================

@pytest.fixture(name="db_session")
def fixture_db_session():
    """Crea una base de datos SQLite en memoria limpia para cada test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,   # NEW
    )
    Base.metadata.create_all(bind=engine)
    
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()

    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(name="repo_service")
def fixture_repo_service(db_session: Session):
    """Instancia del servicio de repositorios respaldada por la sesión de pruebas."""
    return RepositoryService(db_session)


# ==============================================================================
# TESTS DE INTEGRACIÓN DE PERSISTENCIA
# ==============================================================================

def test_persist_flux_multi_checkpoint_repository(repo_service: RepositoryService, db_session: Session):
    """
    Verifica que la estructura de un repositorio tipo FLUX (múltiples checkpoints)
    se persista correctamente en las tablas relacionales.
    """
    repo_path = DATA_DIR / "image/black-forest-labs__FLUX.1-dev"
    if not repo_path.exists():
        pytest.skip(f"Fixture no encontrada en: {repo_path}")

    # Ejecutar el servicio de persistencia
    repo_record = repo_service.process_and_persist(repo_path)

    assert repo_record is not None
    assert repo_record.id == "black-forest-labs/FLUX.1-dev"

    # Consultar desde la BD para asegurar que se guardó relacionalmente
    db_repo = db_session.query(RepositoryModel).filter_by(id="black-forest-labs/FLUX.1-dev").first()
    assert db_repo is not None
    assert len(db_repo.units) > 0

    # Verificar unidades y artefactos directos
    units = db_session.query(ModelUnitModel).filter_by(repo_id=db_repo.id).all()
    assert len(units) >= 1

    artifacts = db_session.query(ArtifactModel).all()
    assert len(artifacts) > 0


def test_idempotent_reprocessing(repo_service: RepositoryService, db_session: Session):
    """
    Verifica que procesar el mismo repositorio dos veces reemplace los registros
    sin duplicar claves primarias ni lanzar integrity errors.
    """
    repo_path = DATA_DIR / "image/black-forest-labs__FLUX.1-dev"
    if not repo_path.exists():
        pytest.skip(f"Fixture no encontrada en: {repo_path}")

    # Primera ejecución
    repo_service.process_and_persist(repo_path)
    count_first = db_session.query(RepositoryModel).count()

    # Segunda ejecución (reprocesado idempotente)
    repo_service.process_and_persist(repo_path)
    count_second = db_session.query(RepositoryModel).count()

    assert count_first == 1
    assert count_second == 1


def test_process_non_existent_repo(repo_service: RepositoryService):
    """Verifica que procesar un directorio inexistente o sin metadatos devuelva None de forma segura."""
    fake_path = Path("/tmp/non_existent_repository_12345")
    result = repo_service.process_and_persist(fake_path)
    assert result is None
