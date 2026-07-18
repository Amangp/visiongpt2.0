import torch
import pickle
from PIL import Image
from torchvision import transforms
from src.model import VQATransformer
from src.model_fusion import VQATransformerFusion
from src.preprocesss import normalize_text

class VQAInference:
    def __init__(self, model_path, word2idx_path, ans2idx_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load vocabularies
        with open(word2idx_path, 'rb') as f:
            self.word2idx = pickle.load(f)
        with open(ans2idx_path, 'rb') as f:
            self.ans2idx = pickle.load(f)
        
        # Create reverse mapping for answers
        self.idx2ans = {v: k for k, v in self.ans2idx.items()}
        
        # Initialize model
        vocab_size = len(self.word2idx)
        ans_size = len(self.ans2idx)

        # Load trained weights (and infer model type)
        checkpoint_obj = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint_obj
        if isinstance(checkpoint_obj, dict) and 'state_dict' in checkpoint_obj and isinstance(checkpoint_obj['state_dict'], dict):
            state_dict = checkpoint_obj['state_dict']

        ckpt_keys = []
        try:
            ckpt_keys = list(state_dict.keys())
        except Exception:
            ckpt_keys = []

        looks_like_fusion_arch = any(k.startswith('fusion_layers.') for k in ckpt_keys) or any(
            k.startswith('vision.patch_embed') for k in ckpt_keys
        )

        if looks_like_fusion_arch:
            # Infer key hyperparameters from the checkpoint tensors.
            vision_dim = int(state_dict['vision.cls_token'].shape[-1])
            patch_size = int(state_dict['vision.patch_embed.weight'].shape[-1])
            pos_len = int(state_dict['vision.pos_embed'].shape[1])
            num_patches = pos_len - 1
            grid = int(num_patches**0.5)
            image_size = grid * patch_size

            vision_depth = 1 + max(
                int(k.split('.')[3])
                for k in ckpt_keys
                if k.startswith('vision.encoder.layers.') and k.split('.')[3].isdigit()
            )
            text_layers = 1 + max(
                int(k.split('.')[3])
                for k in ckpt_keys
                if k.startswith('text.encoder.layers.') and k.split('.')[3].isdigit()
            )
            fusion_layers = 1 + max(
                int(k.split('.')[1])
                for k in ckpt_keys
                if k.startswith('fusion_layers.') and k.split('.')[1].isdigit()
            )

            embed_dim = int(state_dict['text.embedding.weight'].shape[1])
            hidden_dim = int(state_dict['text.proj.weight'].shape[0])
            max_question_len = int(state_dict['text.pos_embedding.weight'].shape[0] - 1)

            # Derive MLP ratio from the trained FFN shapes.
            fusion_ff_dim = int(state_dict['fusion_layers.0.q_ff.net.0.weight'].shape[0])
            fusion_mlp_ratio = fusion_ff_dim / float(hidden_dim)

            self.model = VQATransformerFusion(
                vocab_size=vocab_size,
                ans_size=ans_size,
                image_size=image_size,
                patch_size=patch_size,
                vision_dim=vision_dim,
                vision_depth=vision_depth,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_heads=12,
                text_layers=text_layers,
                fusion_layers=fusion_layers,
                fusion_mlp_ratio=fusion_mlp_ratio,
                dropout=0.0,
                max_question_len=max_question_len,
            )

            self.max_question_len = max_question_len
        else:
            self.model = VQATransformer(
                vocab_size=vocab_size,
                ans_size=ans_size,
                embed_dim=256,
                hidden_dim=512,
                num_heads=8
            )
            self.max_question_len = 20

        try:
            self.model.load_state_dict(state_dict)
        except Exception as e:
            looks_like_alt_arch = any(k.startswith('fusion_layers.') for k in ckpt_keys) or any(k.startswith('vision.patch_embed') for k in ckpt_keys)
            classifier_weight = None
            try:
                classifier_weight = state_dict.get('classifier.4.weight')
            except Exception:
                classifier_weight = None

            extra_hint = ""
            if looks_like_alt_arch:
                extra_hint = (
                    "\n\nThis checkpoint looks like a different architecture (e.g. it contains keys like "
                    "'fusion_layers.*' / 'vision.patch_embed.*'), which is not compatible with the "
                    "current model in src/model.py."
                )
                if hasattr(classifier_weight, 'shape'):
                    extra_hint += f"\nIt also appears to have an output classifier of shape {tuple(classifier_weight.shape)}, while this app expects {ans_size} answers."

            raise RuntimeError(
                f"Failed to load checkpoint '{model_path}'. "
                f"Make sure the checkpoint was trained with the same code + vocab files (word2idx/ans2idx)."
                f"\nOriginal error: {e}{extra_hint}"
            ) from e
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Model loaded successfully on {self.device}")
        print(f"📊 Vocabulary size: {vocab_size}")
        print(f"📊 Answer vocabulary size: {ans_size}")
    
    def preprocess_question(self, question, max_len=None):
        """Convert question text to tensor of word indices."""
        if max_len is None:
            max_len = self.max_question_len
        question = normalize_text(question)
        words = question.split()[:max_len]
        indices = [self.word2idx.get(w, self.word2idx['<unk>']) for w in words]
        
        # Pad to max_len
        if len(indices) < max_len:
            indices += [self.word2idx['<pad>']] * (max_len - len(indices))
        
        return torch.tensor(indices, dtype=torch.long)
    
    def preprocess_image(self, image):
        """Convert PIL image to tensor."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.transform(image)
    
    def predict(self, image, question, top_k=5):
        """
        Predict answer for given image and question.
        
        Args:
            image: PIL Image or path to image
            question: str, question about the image
            top_k: int, number of top answers to return
            
        Returns:
            list of tuples (answer, confidence)
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image)
        
        # Preprocess
        img_tensor = self.preprocess_image(image).unsqueeze(0).to(self.device)
        ques_tensor = self.preprocess_question(question).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(img_tensor, ques_tensor)
            probs = torch.softmax(logits, dim=1)
            top_k = min(top_k, probs.shape[1])
            top_probs, top_indices = torch.topk(probs, top_k, dim=1)
        
        # Convert to answers
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            answer = self.idx2ans.get(idx.item(), '<unk>')
            confidence = prob.item()
            results.append((answer, confidence))
        
        return results

if __name__ == "__main__":
    # Example usage
    vqa = VQAInference(
        model_path='vqa_transformer.pth',
        word2idx_path='word2idx.pkl',
        ans2idx_path='ans2idx.pkl'
    )
    
    # Test with a sample image
    # image_path = "path/to/your/image.jpg"
    # question = "What is in the image?"
    # predictions = vqa.predict(image_path, question)
    # print(f"\nQuestion: {question}")
    # print("Top predictions:")
    # for ans, conf in predictions:
    #     print(f"  {ans}: {conf:.2%}")
