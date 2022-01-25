# eap-aegis
https://www.aegis-info.com/home

## Prerequisites

```
pip install -r requirements.txt
```

## Flow

![index flow](.github/index.svg)

## Usage


Index the data at `toy-data/case_parse_1234.json`

```
python app.py -f toy-data/case_parse_1234.json
```

Open [http://localhost:45678/docs](http://localhost:45678/docs) in your brower. Enter the query

```json
{
  "data": [
    {"text": "交通肇事"
    }
  ],
  "parameters": {
    "traversal_paths": ["r"]
  }
}
```

Or use the `curl`,

```bash
curl --request POST -d '{"parameters":{"traversal_paths": ["r", "c"]}, "data": ["text": "交通肇事"]}' -H 'Content-Type: application/json' 'http://localhost:45678/search'
```
![](.github/restful.png)
![](.github/restful.gif)