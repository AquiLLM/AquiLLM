
## Running the project

### Docker development
Run `docker compose -f deploy/compose/no_gpu_dev.yml --profile web up` to start everything including the web container.
Run commands on the project with `docker compose -f deploy/compose/no_gpu_dev.yml exec web <command>`.

### Credentials
The username and password for the dev superuser are `dev` and `rickbailey` respectively.
You can use the playwright MCP server to test features, which I expect you to do when you make changes that could affect functionality.

### E2E tests
Run E2E tests with `docker compose -f deploy/compose/no_gpu_dev.yml exec web pytest -m e2e -v`.
Run E2E tests in parallel with `docker compose -f deploy/compose/no_gpu_dev.yml exec web pytest -m e2e -n auto`.
Exclude E2E tests when running unit/integration tests with `pytest -m "not e2e"`.
E2E tests live in `aquillm/tests/e2e/` and use the `@pytest.mark.e2e` marker.
