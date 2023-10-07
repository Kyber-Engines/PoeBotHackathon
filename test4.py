import os
import tempfile

import modal

stub = modal.Stub()
pygithub_image = modal.Image.debian_slim().pip_install("PyGithub>=1.59")
git_image = modal.Image.debian_slim().apt_install("git").pip_install("GitPython")


@stub.function(image=pygithub_image, secret=modal.Secret.from_name("my-github-secret"))
def get_username():
    import github

    g = github.Github(auth=github.Auth.Token(os.environ["GITHUB_TOKEN"]))
    return g.get_user().login


@stub.function(image=git_image, secret=modal.Secret.from_name("my-github-secret"))
def clone_repo(repo_url, branch="main"):
    import git

    assert repo_url.startswith("https://")
    repo_url_with_creds = repo_url.replace("https://", "https://" + os.environ["GITHUB_TOKEN"] + "@")
    with tempfile.TemporaryDirectory() as dname:
        print("Cloning", repo_url, "to", dname)
        git.Repo.clone_from(repo_url_with_creds, dname, branch=branch)
        return os.listdir(dname)


@stub.local_entrypoint()
def main(repo_url: str):
    # Run this script with a repo url passed as a command line option
    # For instance `python github_clone_repo.py https://github.com/myuser/myrepo`
    # The repo can be private
    print("Github username:", get_username.remote())
    print("Repo files:", clone_repo.remote(repo_url))
