import argparse
import json
import traceback
import sys
from io import BytesIO
from pathlib import Path

from document_converter.schema import ConversionResult
from document_converter.service import DoclingDocumentConversion


def _bool_arg(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "y", "on")


def main() -> int:
    parser = argparse.ArgumentParser(description="Docling conversion worker (process-isolated).")

    parser.add_argument("--mode", choices=["single", "batch", "serve"], required=True)
    parser.add_argument("--extract-tables", type=_bool_arg, default=False)
    parser.add_argument("--include-images", type=_bool_arg, default=False)
    parser.add_argument("--image-scale", type=int, default=4)

    # single
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--filename", type=str)

    # batch
    parser.add_argument("--batch-json-path", type=str)

    args = parser.parse_args()

    converter = DoclingDocumentConversion()

    if args.mode == "serve":
        # Persistent mode: read one JSON request per line from stdin, write one JSON response per line to stdout.
        # Request format examples:
        # {"mode":"single","input_path":"/tmp/x.pdf","filename":"x.pdf","extract_tables":false,"include_images":false,"image_scale":4}
        # {"mode":"batch","batch":[{"path":"/tmp/a.pdf","filename":"a.pdf"}], ...}
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                rmode = req.get("mode")
                extract_tables = bool(req.get("extract_tables", False))
                include_images = bool(req.get("include_images", False))
                image_scale = int(req.get("image_scale", 4))

                if rmode == "single":
                    p = Path(req["input_path"])
                    filename = req["filename"]
                    res = converter.convert(
                        (filename, BytesIO(p.read_bytes())),
                        extract_tables=extract_tables,
                        include_images=include_images,
                        image_resolution_scale=image_scale,
                    )
                    payload = res.model_dump() if hasattr(res, "model_dump") else res.dict()
                    sys.stdout.write(json.dumps({"ok": True, "result": payload}) + "\n")
                    sys.stdout.flush()
                    continue

                if rmode == "batch":
                    batch = req["batch"]
                    documents = []
                    for item in batch:
                        p = Path(item["path"])
                        documents.append((item["filename"], BytesIO(p.read_bytes())))
                    results = converter.convert_batch(
                        documents,
                        extract_tables=extract_tables,
                        include_images=include_images,
                        image_resolution_scale=image_scale,
                    )
                    payload = [
                        (r.model_dump() if hasattr(r, "model_dump") else r.dict())  # type: ignore[attr-defined]
                        for r in results
                    ]
                    sys.stdout.write(json.dumps({"ok": True, "result": payload}) + "\n")
                    sys.stdout.flush()
                    continue

                sys.stdout.write(json.dumps({"ok": False, "error": f"Unknown mode: {rmode}"}) + "\n")
                sys.stdout.flush()
            except Exception as e:
                sys.stdout.write(
                    json.dumps(
                        {
                            "ok": False,
                            "error": str(e),
                            "trace": traceback.format_exc(limit=20),
                        }
                    )
                    + "\n"
                )
                sys.stdout.flush()
        return 0

    if args.mode == "single":
        if not args.input_path or not args.filename:
            raise SystemExit("--input-path and --filename are required for --mode single")

        p = Path(args.input_path)
        file_bytes = p.read_bytes()
        res = converter.convert(
            (args.filename, BytesIO(file_bytes)),
            extract_tables=args.extract_tables,
            include_images=args.include_images,
            image_resolution_scale=args.image_scale,
        )
        payload = res.model_dump() if hasattr(res, "model_dump") else res.dict()
        sys.stdout.write(json.dumps(payload))
        return 0

    # batch
    if not args.batch_json_path:
        raise SystemExit("--batch-json-path is required for --mode batch")

    batch = json.loads(Path(args.batch_json_path).read_text())
    documents = []
    for item in batch:
        p = Path(item["path"])
        documents.append((item["filename"], BytesIO(p.read_bytes())))

    results = converter.convert_batch(
        documents,
        extract_tables=args.extract_tables,
        include_images=args.include_images,
        image_resolution_scale=args.image_scale,
    )
    payload = [
        (r.model_dump() if hasattr(r, "model_dump") else r.dict())  # type: ignore[attr-defined]
        for r in results
    ]
    sys.stdout.write(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


