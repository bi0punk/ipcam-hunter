#!/usr/bin/env python3
import argparse
import requests

def main():
    ap = argparse.ArgumentParser(description="Lista grupos desde waha para obtener chat_id (@g.us)")
    ap.add_argument("--waha", default="http://127.0.0.1:3000", help="Base URL de waha")
    ap.add_argument("--session", default="default", help="Nombre de sesión")
    ap.add_argument("--api-key", default="", help="X-Api-Key (si aplica)")
    ap.add_argument("--limit", type=int, default=100, help="limit")
    ap.add_argument("--offset", type=int, default=0, help="offset")
    args = ap.parse_args()

    url = f"{args.waha.rstrip('/')}/api/{args.session}/groups"
    headers = {}
    if args.api_key:
        headers["X-Api-Key"] = args.api_key

    params = {"limit": args.limit, "offset": args.offset, "sortBy": "subject", "sortOrder": "asc", "exclude": "participants"}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()

    groups = r.json()
    # La respuesta varía por engine, pero suele incluir id + subject/name
    print("=== Grupos ===")
    for g in groups:
        gid = g.get("id") or g.get("_id") or g.get("wid") or "?"
        subject = g.get("subject") or g.get("name") or g.get("title") or ""
        print(f"- {subject}  ->  {gid}")

if __name__ == "__main__":
    main()
