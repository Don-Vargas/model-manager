import hashlib

import boto3

from .config import settings


s3 = boto3.client(
    "s3",
    endpoint_url=settings.minio_endpoint,
    aws_access_key_id=settings.minio_access_key,
    aws_secret_access_key=settings.minio_secret_key,
    region_name="us-east-1",
)

def build_artifact_prefix(
    provider: str,
    source_id: str,
    revision: str,
    artifact_format: str,
) -> str:
    return (
        f"{provider}/"
        f"{source_id}/"
        f"{revision}/"
        f"{artifact_format}/"
    )


def object_exists(key: str) -> bool:
    try:
        s3.head_object(
            Bucket=settings.minio_bucket,
            Key=key,
        )
        return True

    except s3.exceptions.ClientError:
        return False


def object_matches_size(
    key: str,
    expected_size: int | None,
) -> bool:

    if expected_size is None:
        return False

    try:
        response = s3.head_object(
            Bucket=settings.minio_bucket,
            Key=key,
        )

        return response["ContentLength"] == expected_size

    except s3.exceptions.ClientError:
        return False


def calculate_object_sha256(
    bucket: str,
    key: str,
) -> str:

    response = s3.get_object(
        Bucket=bucket,
        Key=key,
    )

    sha256 = hashlib.sha256()

    body = response["Body"]

    try:
        while True:
            chunk = body.read(8 * 1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    finally:
        body.close()

    return sha256.hexdigest()

def upload_file(
    local_path: str,
    key: str,
) -> None:
    s3.upload_file(
        local_path,
        settings.minio_bucket,
        key,
    )

