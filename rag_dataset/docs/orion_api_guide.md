# Orion Notes API Guide

## Base URL
https://api.orion-notes.example/v1

## Authentication
Use an API token in the Authorization header:
Authorization: Bearer <token>

## Endpoints
### Create note
POST /notes
Body:
{
  "title": "string",
  "content": "string",
  "tags": ["string"]
}

### List notes
GET /notes
Query params:
- tag: filter by a single tag
- limit: max results (default 20, max 100)

### Get note
GET /notes/{id}

### Delete note
DELETE /notes/{id}

## Rate Limits
- 60 requests per minute per token
- Burst limit: 10 requests per second

## Errors
- 401 Unauthorized: missing or invalid token
- 404 Not Found: note id does not exist
- 429 Too Many Requests: rate limit exceeded
