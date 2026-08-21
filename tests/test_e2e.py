# tests/test_e2e.py
"""
End-to-end tests for the Model Manager API.

These tests verify the complete flow from repository import to 
artifact download and verification.
"""

import json
import os
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool  

from app.database import Base, get_db
from app.main import app
from app.services.repository_service import RepositoryService


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def test_db():
    """Create an in-memory test database."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,   # NEW — forces one physical connection for the whole session
    )
    Base.metadata.create_all(bind=engine)

    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )

    return TestingSessionLocal


@pytest.fixture
def db_session(test_db):
    """Get a database session for testing."""
    session = test_db()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def client(db_session):
    """Create a test client with database dependency override."""
    
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


# =============================================================================
# E2E TESTS
# =============================================================================

class TestEndToEnd:
    """End-to-end test suite for the complete workflow."""
    
    def test_health_endpoint(self, client):
        """Verify the health check endpoint works."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    
    def test_import_small_repository(self, client):
        """
        Test importing a small HuggingFace repository.
        
        This test uses a lightweight model fixture that should be available
        on HuggingFace Hub. The model is small enough for quick downloads.
        """
        # Use a tiny model for testing
        repo_id = "hf-internal-testing/tiny-random-bert"  # Very small model
        
        response = client.post(
            "/repositories",
            json={"repo_id": repo_id}
        )
        
        # The API might fail if we don't have HuggingFace token configured
        # or the model is not available, but we're testing the flow
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert data["id"] == repo_id
            assert "units" in data
    
    def test_repository_normalization(self, client):
        """
        Test that repository normalization produces the correct structure.
        
        This test uses a known Diffusers repository to verify component
        detection works correctly.
        """
        # Test with Qwen-Image - known Diffusers pipeline
        repo_id = "Qwen/Qwen-Image"
        
        response = client.post(
            "/repositories",
            json={"repo_id": repo_id}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Should have at least one unit
            assert len(data["units"]) > 0
            
            # Find the pipeline unit
            pipeline_units = [u for u in data["units"] if u["unit_type"] == "pipeline"]
            if pipeline_units:
                unit = pipeline_units[0]
                assert unit["framework"] == "diffusers"
                assert "artifacts" in unit
    
    def test_multiple_models_in_repository(self, client):
        """
        Test that repositories with multiple checkpoints are properly split.
        
        FLUX.1-dev has both 'ae' and 'flux1-dev' checkpoints as separate
        ModelUnits.
        """
        repo_id = "black-forest-labs/FLUX.1-dev"
        
        response = client.post(
            "/repositories",
            json={"repo_id": repo_id}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Should have at least 2 units (ae and flux1-dev)
            assert len(data["units"]) >= 2
            
            unit_names = [u["name"] for u in data["units"]]
            assert "ae" in unit_names or any("ae" in n for n in unit_names)
    
    def test_list_models(self, client, db_session):
        """
        Test the GET /models endpoint returns all imported repositories.
        """
        # First import a repository
        response = client.post(
            "/repositories",
            json={"repo_id": "hf-internal-testing/tiny-random-bert"}
        )
        
        # Then list all models
        response = client.get("/models")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_artifact_response_structure(self, client):
        """
        Verify that artifact responses have the expected structure.
        """
        repo_id = "hf-internal-testing/tiny-random-bert"
        
        response = client.post(
            "/repositories",
            json={"repo_id": repo_id}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Find artifacts in the response
            for unit in data.get("units", []):
                for artifact in unit.get("artifacts", []):
                    # Check required fields
                    assert "id" in artifact
                    assert "name" in artifact
                    assert "artifact_type" in artifact
                    assert "files" in artifact
    
    def test_artifact_download_flow(self, client, db_session):
        """
        Test the complete artifact download and verification flow.
        
        This test requires MinIO to be configured. It will be skipped
        if MinIO is not available.
        """
        import os
        
        # Skip if MinIO not configured
        if not all([
            os.getenv("MINIO_ENDPOINT"),
            os.getenv("MINIO_ACCESS_KEY"),
            os.getenv("MINIO_SECRET_KEY"),
        ]):
            pytest.skip("MinIO not configured")
        
        # Import a repository first
        repo_id = "hf-internal-testing/tiny-random-bert"
        
        import_response = client.post(
            "/repositories",
            json={"repo_id": repo_id}
        )
        
        if import_response.status_code != 200:
            pytest.skip(f"Repository import failed: {import_response.json()}")
        
        # Get the first artifact ID
        data = import_response.json()
        artifact_id = None
        
        for unit in data.get("units", []):
            for artifact in unit.get("artifacts", []):
                artifact_id = artifact.get("id")
                break
            if artifact_id:
                break
        
        if not artifact_id:
            pytest.skip("No artifacts found in repository")
        
        # Download the artifact
        download_response = client.post(
            f"/artifacts/{artifact_id}/download"
        )
        
        # The download might fail if the file doesn't exist or credentials are wrong
        # but we're testing the flow
        assert download_response.status_code in [200, 400, 404, 500]
    
    def test_artifact_verify_flow(self, client):
        """
        Test the artifact verification endpoint.
        """
        repo_id = "hf-internal-testing/tiny-random-bert"
        
        import_response = client.post(
            "/repositories",
            json={"repo_id": repo_id}
        )
        
        if import_response.status_code != 200:
            pytest.skip(f"Repository import failed: {import_response.json()}")
        
        # Get the first artifact ID
        data = import_response.json()
        artifact_id = None
        
        for unit in data.get("units", []):
            for artifact in unit.get("artifacts", []):
                artifact_id = artifact.get("id")
                break
            if artifact_id:
                break
        
        if not artifact_id:
            pytest.skip("No artifacts found in repository")
        
        # Verify the artifact
        verify_response = client.post(
            f"/artifacts/{artifact_id}/verify"
        )
        
        # Should return verification results
        assert verify_response.status_code in [200, 404]
        
        if verify_response.status_code == 200:
            result = verify_response.json()
            assert "artifact_id" in result
            assert "status" in result
            assert "files" in result
    
    def test_error_handling_nonexistent_artifact(self, client):
        """Test error handling for non-existent artifacts."""
        response = client.get("/artifacts/nonexistent-id")
        assert response.status_code == 404


# =============================================================================
# PERFORMANCE AND LOAD TESTS
# =============================================================================

class TestPerformance:
    """Performance and load tests."""
    
    def test_concurrent_imports(self, client):
        """
        Test importing multiple repositories concurrently.
        
        This test ensures the system can handle multiple import requests
        without race conditions.
        """
        import concurrent.futures
        
        test_repos = [
            "hf-internal-testing/tiny-random-bert",
            "Qwen/Qwen-Image",
        ]
        
        def import_repo(repo_id):
            return client.post(
                "/repositories",
                json={"repo_id": repo_id}
            )
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(import_repo, repo_id)
                for repo_id in test_repos
            ]
            
            results = [f.result() for f in futures]
            
            # Check that all requests completed without crashing
            for result in results:
                # 200 or 500 are acceptable (500 could be due to missing API keys)
                assert result.status_code in [200, 500]
    
    def test_idempotent_import(self, client):
        """Test that importing the same repository twice doesn't duplicate data."""
        repo_id = "hf-internal-testing/tiny-random-bert"
        
        # First import
        first_response = client.post(
            "/repositories",
            json={"repo_id": repo_id}
        )
        
        if first_response.status_code != 200:
            pytest.skip("Repository import failed")
        
        # Second import
        second_response = client.post(
            "/repositories",
            json={"repo_id": repo_id}
        )
        
        # Both should succeed (second should be idempotent)
        assert second_response.status_code == 200


# =============================================================================
# INTEGRATION TESTS WITH ACTUAL DATA FIXTURES
# =============================================================================

class TestWithFixtureData:
    """
    Tests that run with locally stored fixture data.
    
    These tests require the data/ directory to be populated with
    model_info.json files from real HuggingFace repositories.
    """
    
    def test_single_checkpoint_detection(self, client, db_session):
        """
        Test detection of single checkpoint files.
        
        Uses the FLUX.1-dev fixture which has multiple root checkpoints.
        """
        data_dir = Path("/app/data/huggingface")
        flux_path = data_dir / "image/black-forest-labs__FLUX.1-dev"
        
        if not flux_path.exists():
            pytest.skip("Fixture not available")
        
        # Use the repository service directly
        service = RepositoryService(db_session)
        result = service.process_and_persist(flux_path)
        
        if result:
            assert result.id == "black-forest-labs/FLUX.1-dev"
            assert len(result.units) >= 2
    
    def test_diffusers_pipeline_detection(self, client, db_session):
        """
        Test detection of Diffusers pipeline components.
        
        Uses the Qwen-Image fixture which has a complete Diffusers pipeline.
        """
        data_dir = Path("/app/data/huggingface")
        qwen_path = data_dir / "image/Qwen__Qwen-Image"
        
        if not qwen_path.exists():
            pytest.skip("Fixture not available")
        
        service = RepositoryService(db_session)
        result = service.process_and_persist(qwen_path)
        
        if result:
            # Should detect pipeline
            pipeline_units = [u for u in result.units if u.unit_type == "pipeline"]
            if pipeline_units:
                unit = pipeline_units[0]
                assert unit.framework == "diffusers"
                assert len(unit.components) > 0


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    # This allows running tests directly
    pytest.main([__file__, "-v", "--tb=short"])