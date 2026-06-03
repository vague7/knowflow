# Developer Onboarding Guide

Welcome to the engineering team. This guide walks you through the essential tools and workflows you need to start contributing code.

## Git Access

All source code is hosted on GitHub Enterprise at github.company.com. Request access via the IT Service Portal under "Application Access" and select "GitHub Enterprise — Developer". Your manager must approve the request. Once granted, configure SSH keys by following the guide in Confluence under "Engineering > SSH Setup". Clone repositories using SSH, not HTTPS. All repos follow the naming convention: team-name/project-name. You will be added to your team's GitHub organization within one business day of approval.

## CI/CD Pipeline

We use Jenkins for continuous integration and ArgoCD for deployments. Every push to a feature branch triggers a CI build that runs linting, unit tests, and security scans. Builds must pass before a pull request can be merged. The CI configuration lives in a Jenkinsfile at the root of each repository. Staging deployments happen automatically when code is merged to the develop branch. Production deployments require a manual approval step in ArgoCD and are only triggered from the main branch. Build status is reported to Slack in the #ci-notifications channel.

## Code Review Process

All changes must go through a pull request with at least two approved reviews before merging. Reviewers should be assigned from your team, and at least one must be a senior engineer. PR descriptions must include a summary of changes, testing steps, and any related Jira ticket numbers. Reviews should be completed within one business day. Use conventional commits for commit messages (e.g., feat:, fix:, docs:). Squash merge is the default merge strategy. After merging, delete the feature branch to keep the repository clean.
