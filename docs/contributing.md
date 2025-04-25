# Instructions for Developers
We would love for you to contribute new code, bugfixes, or documentation! 

## Development Workflow
We use a git-based [_feature branch_ workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow#:~:text=The%20core%20idea%20behind%20the,without%20disturbing%20the%20main%20codebase.). This means that new development should happen on a feature-specific branch, which will then be merged in to the `main` branch via a pull request that includes a peer-review of the feature code.

Here is an outline of the workflow:

1. Check out the `main` branch and ensure it is up-to-date with the remote repository
```
git checkout main
git fetch origin
```

2. Create a new branch for your feature:
```
git checkout -b my-awesome-feature
```

3. Develop your feature on the feature branch, using some combination of commands like
```
git status
git add <some-file>
git commit -m "add some-file"
```
at this stage you can also `git push` your branch to the remote repository if you want others to see and/or collaborate on your branch, or you can keep it local.

4. Once you have finished developing your feature, you should perform some housekeeping to ensure your code conforms to standards:

> Note: if you have access to `make` on your system, you can run the following commands via the shortcuts `make format`, `make lint`, and `make test` (see `Makefile`)

- #### Ensure that your branch is up-to-date with main:
Other developers may have committed changes to `main` since you started your branch. You should make sure you pull in those changes to avoid merge conflicts. To do this, from your feature branch run:
```
git merge main
```

- #### Format your code:
These tools will automatically format your code files to match our chosen formatting standard.
```
black tumortwin
isort --profile black tumortwin
```

   - #### Lint your code:
These tools will run some static code analysis tools to point out possible issues or improvements. You should try to fix as many of the issues flagged as possible. If there is something that you aren't sure about, feel free to note it in a comment in your pull request (next step)
```
flake8 tumortwin
mypy tumortwin
```
   - #### Run automated tests
This will run automated tests that we have written, to ensure that your feature hasn't broken some functionality elsewhere. You should also add tests for your new feature!
```
pytest tests
```

5\. Request that your feature branch be merged in to the `main` branch by opening a pull-request.

Navigate to the [Pull Requests](https://github.com/OncologyModelingGroup/TumorTwin/pulls) tab on github. Click `New pull request` and follow the instructions. You should assign yourself to the pull request, and you can request a code review by assigning a reviewer (Alternatively, post on the group slack to say that a feature is ready for review!) 

6\. Address Code Review comments

The code reviewer may make ask questions or suggest improvements via comments on the pull request. Simply address the comments via new commits to your feature branch

7\. Merge your feature!

Hit merge in the pull request on github. `Merge and Squash` is the prefered method, as this will squash all of your feature commits into a single commit, keeping the `main` branch history clean.