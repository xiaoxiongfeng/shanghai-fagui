# eap-aegis
https://www.aegis-info.com/home

## Prerequisites

```
pip install -r requirements.txt
```

## Index Flow

![index flow](.github/index.svg)

## Query Flow
![query flow](.github/query.svg)


## Usage

For demonstration purpose, we use the ERNIE model to encode the `_source.title`. The same field is used for both 
indexing and querying


![](.github/doc.svg)

### Index & Query

Index the data at `toy-data/case_parse_10.json`

```
python app.py -f toy-data/case_parse_1234.json
```

Open [http://localhost:45678/docs](http://localhost:45678/docs) in your brower.

![](.github/restful.png)
![](.github/restful.gif)