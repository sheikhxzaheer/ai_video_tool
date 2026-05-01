import os

# 👉 Put your txt file path here
TXT_FILE_PATH = r"C:\Nil AI-ML\run_index\Temp\faild.txt"

def delete_files_from_txt(txt_path):
    if not os.path.exists(txt_path):
        print(f"❌ TXT file not found: {txt_path}")
        return

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()

    deleted_count = 0
    not_found_count = 0

    for line in lines:
        file_path = line.strip()

        if not file_path:
            continue  # skip empty lines

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"✅ Deleted: {file_path}")
                deleted_count += 1
            else:
                print(f"⚠️ Not found: {file_path}")
                not_found_count += 1
        except Exception as e:
            print(f"❌ Error deleting {file_path}: {e}")

    print("\n--- Summary ---")
    print(f"Deleted files: {deleted_count}")
    print(f"Not found: {not_found_count}")

# Run
delete_files_from_txt(TXT_FILE_PATH)