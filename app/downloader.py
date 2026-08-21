# app/downloader.py
import hashlib
import httpx
from .providers.huggingface import HuggingFaceProvider
from .storage import (
    s3,
    settings,
    object_matches_size,
)

CHUNK_SIZE = 64 * 1024 * 1024  # 64 MB


class DownloadError(Exception):
    pass


def download_file_to_minio(
    provider: HuggingFaceProvider,
    repo_id: str,
    revision: str | None,
    filename: str,
    object_key: str,
    expected_size: int | None = None,
    expected_checksum: str | None = None,
):
    if object_matches_size(object_key, expected_size):
        print(f"Skipping {filename}: already exists in MinIO")
        return {
            "status": "skipped",
            "key": object_key,
            "size": expected_size,
            "checksum": expected_checksum,
        }

    url = provider.get_file_url(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
    )

    headers = {}
    if provider.token:
        headers["Authorization"] = f"Bearer {provider.token}"

    upload = s3.create_multipart_upload(
        Bucket=settings.minio_bucket,
        Key=object_key,
    )

    upload_id = upload["UploadId"]
    parts = []
    part_number = 1

    sha256 = hashlib.sha256()
    downloaded_bytes = 0

    try:
        with httpx.Client(follow_redirects=True, timeout=None) as client:
            with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()

                # Iteramos pidiendo bloques de hasta CHUNK_SIZE directamente a la red.
                # httpx pausará la lectura de red (backpressure) mientras se ejecuta s3.upload_part.
                for part_data in response.iter_bytes(chunk_size=CHUNK_SIZE):
                    if not part_data:
                        continue

                    sha256.update(part_data)
                    downloaded_bytes += len(part_data)

                    result = s3.upload_part(
                        Bucket=settings.minio_bucket,
                        Key=object_key,
                        PartNumber=part_number,
                        UploadId=upload_id,
                        Body=part_data,
                    )

                    parts.append(
                        {
                            "PartNumber": part_number,
                            "ETag": result["ETag"],
                        }
                    )

                    print(f"Uploaded part {part_number} of {filename}")
                    part_number += 1

        actual_checksum = sha256.hexdigest()

        print(f"Downloaded {filename}: {downloaded_bytes} bytes")
        print(f"SHA256: {actual_checksum}")

        if expected_size is not None and downloaded_bytes != expected_size:
            raise DownloadError(
                f"Size mismatch for {filename}: expected {expected_size}, got {downloaded_bytes}"
            )

        if expected_checksum is not None and actual_checksum != expected_checksum:
            raise DownloadError(
                f"Checksum mismatch for {filename}: expected {expected_checksum}, got {actual_checksum}"
            )

        s3.complete_multipart_upload(
            Bucket=settings.minio_bucket,
            Key=object_key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )

        return {
            "status": "downloaded",
            "key": object_key,
            "size": downloaded_bytes,
            "checksum": actual_checksum,
        }

    except Exception:
        s3.abort_multipart_upload(
            Bucket=settings.minio_bucket,
            Key=object_key,
            UploadId=upload_id,
        )
        raise