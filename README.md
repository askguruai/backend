# Backend Service

[![Actions Status](https://github.com/askguruai/backend/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/askguruai/backend/actions)

Backend glues together document indexing, Q&A, ranking and reactions. Check [part of the blog](https://asmirnov.xyz/askguru#tech-details) to learn how components interact with each other.

## Deployment

Specify all envs in `.env` and start everything:

```bash
sudo -E docker compose up -d --build
```
