# Backend
Backend service is responsible for handling user requests, orchestrating ML 
components, and providing caching and other layers.

## API
API reference available at [swagger](http://78.141.213.164:5050/docs)

### POST /text_query
Given a raw text and a query (question), returns an answer to this question
#### Examples

<details>
<summary>Create an embedding for single text.</summary>

```bash
curl -X 'POST' \
  'http://78.141.213.164:5555/text_query/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text_input": "%THE WHOLE WIKI ARTICLE ABOUT 
  UKRAIN MILITAARY CONFLICT GOES HERE%"
  "query": "What exactly happen?"
}'
```
Response:
```json
{
  "data": " The invasion began on the morning of 24 February 2022, ......"
}
```
</details>

## Development
1. Clone repo
  ```bash
  git clone git@github.com:askaye/backend.git
  cd ./backend
  ```

2. Install dependencies
  ```bash
  conda create --name backend python=3.10
  conda activate backend
  pip install -r requirements.txt
  ```

3. Run service
  ```bash
  python main.py
```

## TODO
* Resolve ToDos :)
* Add an error-handling wrapper so to prevent 500-codes


  