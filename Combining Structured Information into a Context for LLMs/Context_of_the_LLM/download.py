import subprocess
import os
import shutil
from pathlib import Path
from src.config import commit, date, version, repo_name, problem_stmt

def run_command(command, cwd=None):
    """Execute a shell command and return the result."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=cwd
        )
        print(f"✓ Command executed: {command}")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed: {command}")
        print(f"Error: {e.stderr}")
        raise

def download_and_checkout_project(repo_url, commit_hash, project_dir=None):
    """
    Download a git repository and checkout to a specific commit.
    
    Args:
        repo_url (str): The git repository URL
        commit_hash (str): The commit hash to checkout
        project_dir (str): Directory name for the project (optional)
    """
    # Extract repo name from URL if project_dir not provided
    if project_dir is None:
        project_dir = repo_url.split('/')[-1].replace('.git', '')
    
    # Get current working directory
    original_cwd = os.getcwd()
    print(f"Starting directory: {original_cwd}")
    
    try:
        # Remove existing directory if it exists
        if os.path.exists(project_dir):
            print(f"Removing existing directory: {project_dir}")
            shutil.rmtree(project_dir)
        
        # Clone the repository
        print(f"Cloning repository: {repo_url}")
        run_command(f"git clone {repo_url}")
        
        # Change to project directory
        project_path = os.path.join(original_cwd, project_dir)
        print(f"Changing to directory: {project_path}")
        os.chdir(project_path)
        print(f"Current directory: {os.getcwd()}")
        
        # Checkout to specific commit
        print(f"Checking out to commit: {commit_hash}")
        run_command(f"git checkout {commit_hash}")
        
        # Verify the checkout
        result = run_command("git rev-parse HEAD")
        current_commit = result.stdout.strip()
        print(f"Current commit: {current_commit}")
        
        if current_commit.startswith(commit_hash):
            print("✓ Successfully checked out to the specified commit")
        else:
            print("⚠ Warning: Current commit doesn't match the expected commit")
        
        return project_path
        
    except Exception as e:
        print(f"Error during project setup: {e}")
        raise
    finally:
        # Return to original directory
        os.chdir(original_cwd)
        print(f"Returned to directory: {os.getcwd()}")

def main():
    """Main function to execute the project download and setup."""
    # Configuration from your config file
    repo_url = f"https://github.com/{repo_name}.git"  # Assuming repo_name contains owner/repo
    
    print("=" * 50)
    print("PROJECT DOWNLOADER")
    print("=" * 50)
    print(f"Repository: {repo_url}")
    print(f"Commit: {commit}")
    print(f"Version: {version}")
    print(f"Date: {date}")
    print(f"Problem Statement: {problem_stmt}")
    print("=" * 50)
    
    try:
        project_path = download_and_checkout_project(repo_url, commit)
        print(f"\n✓ Project successfully set up at: {project_path}")
        
        # Navigate to content directory if it exists
        content_dir = os.path.join(project_path, "content")
        if os.path.exists(content_dir):
            print(f"Content directory found: {content_dir}")
        else:
            print("Content directory not found in the project")
            
    except Exception as e:
        print(f"\n✗ Failed to set up project: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()