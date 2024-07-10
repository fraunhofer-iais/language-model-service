# Language Model Service
This repo provides language models via a (REST) endpoint.

## Installation

## Setup the Repository

```bash
git clone git@github.com:fraunhofer-iais/language-model-service.git
cd language-model-service
git checkout master
```
## Running the Service
```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Set environment variables
```bash
export HF_TOKEN=xxx
export OPENAI_API_KEY=xxx
export ALEPH_ALPHA_TOKEN=xxx
```

Now you can start the service using the command 
```bash
python -m src.start
```

## REST API description
When running the service, there is a REST API description available at http://0.0.0.0:<PORT>/docs, where <PORT> is specified in the config file.

## Using REST end-point

## Endpoints

### 1. Generate

**Endpoint:** `/generate`

**Method:** POST

**Description:** Generates token strings based on input strings.

**Request Body:**
```json
[
  "hi"
]
```

**Response Body:**
```json
[
  ", who was born in the city of Kolk"
]
```

### 2. Vectorize

**Endpoint:** `/vectorize`

**Method:** POST

**Description:** Generates embeddings for input strings.

**Request Body:**
```json
[
  "hi"
]
```

**Response Body:**
```json
[[
  0.003456,
  -0.00402,
  ...
]]
```

### 3. Available Models

**Endpoint:** `/available_models`

**Method:** GET

**Description:** Provides description of available models.

**Response Body:**
```json
{
  "gpt-2": {
    "model_provider": "HuggingFace",
    "model": "gpt2",
    "model_type": "AutoModelForCausalLM",
    "tokenizer": "gpt2",
    "use_fast": false,
    "change_pad_token": true,
    "adapter": null,
    "device": "cpu",
    "cache_dir": null,
    "use_accelerate": false
  }
}
```