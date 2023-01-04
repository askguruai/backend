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

### MongoDB

To launch MongoDB container w/ mongo-express UI, do:
  ```bash
  cd ./db
  sudo docker-compose up -d
  ```

To connect to mongo-express forward ports and visit [localhost:8081](http://localhost:8081/):
  ```bash
  ssh -NL 8081:localhost:8081 denpasar
  ```


### Repo

1. Install dependencies
  ```bash
  conda create --name backend python=3.10
  conda activate backend
  pip install -r requirements.txt
  python <<HEREDOC
  import nltk
  nltk.download('punkt')
  HEREDOC
  ```

2. Run service
  ```bash
  python main.py
  ```

## TODO
* Resolve ToDos :)
* Add an error-handling wrapper so to prevent 500-codes
