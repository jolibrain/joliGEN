# JoliGEN server

## Start training

Creates a training process with name `name`

- *Method* `POST`
- *URL* `/train/:name`
- *URL Params*
    - `name:string`: name of training process 
- *Response*
    - *Success:*:
        - Code: 201
        - Content: `{ "message": "ok", "name": "train_1" }`

## Get all training processes

`GET /train`

## Get training status

`GET /train/name`

## Stop training process

`DELETE /train/name`
