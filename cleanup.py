import glob
import os


# Function to list files matching the pattern and delete them
def delete_cache_arrow_files(root_dir):
    # Iterate over all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the file matches the pattern "cache-*.arrow"
            if filename.startswith("cache-") and filename.endswith(".arrow"):
                # Print the file path before deleting
                file_path = os.path.join(dirpath, filename)
                print("Deleting:", file_path)
                # Delete the file
                os.remove(file_path)


# Main function
def main():
    root_dir = "."  # Set the root directory
    print("Files to be deleted:")
    # List files to be deleted
    delete_cache_arrow_files(root_dir)


# Execute the main function
if __name__ == "__main__":
    main()
