from pathlib import Path


def main() -> None:
    folder_input = input("Cartella contenente i file: ").strip().strip('"')
    new_prefix = input("Nuovo prefisso: ").strip()

    folder = Path(folder_input).expanduser().resolve()

    if not folder.is_dir():
        print(f"Errore: la cartella non esiste: {folder}")
        return

    if not new_prefix:
        print("Errore: il prefisso non può essere vuoto.")
        return

    if "/" in new_prefix or "\\" in new_prefix:
        print("Errore: il prefisso non può contenere separatori di percorso.")
        return

    renamed = 0
    skipped = 0

    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue

        # Ignora i file senza underscore.
        if "_" not in file_path.name:
            continue

        _, remaining_name = file_path.name.split("_", 1)
        new_path = file_path.with_name(f"{new_prefix}_{remaining_name}")

        if new_path == file_path:
            continue

        if new_path.exists():
            print(f"Saltato: {new_path.name} esiste già.")
            skipped += 1
            continue

        file_path.rename(new_path)
        print(f"{file_path.name} -> {new_path.name}")
        renamed += 1

    print(f"\nFile rinominati: {renamed}")
    print(f"File saltati: {skipped}")


if __name__ == "__main__":
    main()