# mathbert_server.py
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model and tokenizer
model = None
tokenizer = None
device = None

def load_mathbert():
    """Load MathBERT model and tokenizer at startup"""
    global model, tokenizer, device
    
    logger.info("="*60)
    logger.info("MathBERT Embedding Server")
    logger.info("="*60)
    logger.info("Loading MathBERT model...")
    
    try:
        # Load MathBERT (you can change the model name if using a different variant)
        model_name = "tbs17/MathBERT"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Set to evaluation mode
        model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = model.to(device)
            logger.info(f"✓ MathBERT loaded on GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device("cpu")
            logger.info("✓ MathBERT loaded on CPU")
        
        logger.info(f"✓ Model: {model_name}")
        logger.info(f"✓ Embedding dimension: {model.config.hidden_size}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Failed to load MathBERT: {e}")
        raise

@app.route('/embed', methods=['POST'])
def embed_text():
    """
    Embed mathematical text using MathBERT
    
    Request JSON:
        {
            "text": "Solve for x: 2x + 5 = 17"
        }
    
    Response JSON:
        {
            "embedding": [0.123, -0.456, ...],  # 768-dimensional vector
            "dimension": 768,
            "tokens": 15
        }
    """
    try:
        # Parse request
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request'}), 400
        
        text = data['text']
        
        if not text or not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        logger.info(f"Encoding text: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embedding
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Use CLS token embedding (first token)
            # This is the standard approach for sentence embeddings with BERT
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            # Alternative: Mean pooling over all tokens
            # attention_mask = inputs['attention_mask']
            # token_embeddings = outputs.last_hidden_state
            # input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            # sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            # sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            # cls_embedding = (sum_embeddings / sum_mask).cpu().numpy()[0]
        
        num_tokens = inputs['input_ids'].shape[1]
        embedding_dim = len(cls_embedding)
        
        logger.info(f"✓ Generated embedding: {embedding_dim}D, {num_tokens} tokens")
        
        return jsonify({
            'embedding': cls_embedding.tolist(),
            'dimension': embedding_dim,
            'tokens': num_tokens
        })
    
    except Exception as e:
        logger.error(f"Error during embedding: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/embed_batch', methods=['POST'])
def embed_batch():
    """
    Embed multiple texts in a batch (more efficient)
    
    Request JSON:
        {
            "texts": ["Problem 1", "Problem 2", ...]
        }
    
    Response JSON:
        {
            "embeddings": [[...], [...], ...],
            "dimension": 768,
            "count": 2
        }
    """
    try:
        data = request.json
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing "texts" field in request'}), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({'error': 'texts must be a non-empty list'}), 400
        
        logger.info(f"Batch encoding {len(texts)} texts")
        
        # Tokenize all texts
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # CLS token for each text
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        embeddings_list = [emb.tolist() for emb in cls_embeddings]
        
        logger.info(f"✓ Generated {len(embeddings_list)} embeddings")
        
        return jsonify({
            'embeddings': embeddings_list,
            'dimension': len(embeddings_list[0]),
            'count': len(embeddings_list)
        })
    
    except Exception as e:
        logger.error(f"Error during batch embedding: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'MathBERT',
        'device': str(device),
        'embedding_dimension': model.config.hidden_size if model else None
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with service info"""
    return jsonify({
        'service': 'MathBERT Embedding Server',
        'version': '1.0.0',
        'endpoints': {
            'POST /embed': 'Embed single text',
            'POST /embed_batch': 'Embed multiple texts',
            'GET /health': 'Health check',
        }
    })

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MathBERT Embedding Server')
    parser.add_argument('--port', type=int, default=8081, help='Port to run server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()
    
    # Load model at startup
    load_mathbert()
    
    # Start server
    logger.info(f"\nStarting server on {args.host}:{args.port}")
    logger.info("Endpoints:")
    logger.info(f"  POST http://localhost:{args.port}/embed")
    logger.info(f"  POST http://localhost:{args.port}/embed_batch")
    logger.info(f"  GET  http://localhost:{args.port}/health")
    logger.info("\nServer ready. Press Ctrl+C to stop.")
    logger.info("="*60 + "\n")
    
    app.run(host=args.host, port=args.port, threaded=True, debug=False)

if __name__ == '__main__':
    main()