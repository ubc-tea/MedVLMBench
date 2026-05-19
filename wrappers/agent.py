import inspect

from model.chat import ChatMetaModel


class AgentMetaWrapper(ChatMetaModel):
    def __init__(self, backbone, args):
        super().__init__(args)

        self.backbone = backbone
        self.name = f"Agent({getattr(backbone, 'name', args.model)})"
        self.model_type = getattr(backbone, "model_type", "")
        self.last_trace = {}

        self._sync_backbone_state()

    def _sync_backbone_state(self):
        self.model = getattr(self.backbone, "model", None)
        self.tokenizer = getattr(self.backbone, "tokenizer", None)
        self.processor = getattr(self.backbone, "processor", None)

        self.image_processor = getattr(self.backbone, "image_processor", None)
        self.image_processor_callable = getattr(self.backbone, "image_processor_callable", None)
        self.prefers_cpu_image_inputs = getattr(self.backbone, "prefers_cpu_image_inputs", False)

    def load_from_pretrained(self, model_path, **kwargs):
        self.backbone.load_from_pretrained(model_path, **kwargs)
        self._sync_backbone_state()

    def load_for_training(self, model_path):
        self.backbone.load_for_training(model_path)
        self._sync_backbone_state()

    def save(self, output_folder, trainer=None):
        if trainer is None:
            return self.backbone.save(output_folder)
        return self.backbone.save(output_folder, trainer=trainer)

    def set_inference_context(self, context=None):
        super().set_inference_context(context)
        if hasattr(self.backbone, "set_inference_context"):
            self.backbone.set_inference_context(context)

    def set_device(self, device):
        super().set_device(device)
        if hasattr(self.backbone, "set_device"):
            self.backbone.set_device(device)
        else:
            self.backbone.device = device

    def get_last_trace(self):
        return self.last_trace

    def reset(self):
        self.last_trace = {}

    def _call_backbone_infer(self, image, qs, image_size=None, temperature=None):
        """Call the backbone with only the kwargs its inference method supports."""
        infer_fn = self.backbone.infer_vision_language

        try:
            signature = inspect.signature(infer_fn)
            parameters = signature.parameters.values()
            accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters)
            accepts_image_size = accepts_var_kwargs or "image_size" in signature.parameters
            accepts_temperature = accepts_var_kwargs or "temperature" in signature.parameters
        except (TypeError, ValueError):
            accepts_image_size = image_size is not None
            accepts_temperature = temperature is not None

        kwargs = {}
        if image_size is not None and accepts_image_size:
            kwargs["image_size"] = image_size
        if temperature is not None and accepts_temperature:
            kwargs["temperature"] = temperature

        try:
            return infer_fn(image, qs, **kwargs)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise

            kwargs.pop("temperature", None)
            return infer_fn(image, qs, **kwargs)

    def infer_vision_language(self, image, qs, image_size=None, temperature=None):
        """Delegate inference to the wrapped backbone and normalize empty outputs."""
        result = self.backbone.infer_vision_language(image, qs, image_size=image_size, temperature=temperature)
        if result is None:
            return ""
        if not isinstance(result, str):
            result = str(result)
        return result.strip()

    def _query_backbone(self, image, prompt, image_size=None, temperature=None):
        """Run inference of backbone VLM to get response"""
        result = self._call_backbone_infer(image, prompt, image_size=image_size, temperature=temperature)
        if result is None:
            return ""
        if not isinstance(result, str):
            result = str(result)
        return result.strip()
