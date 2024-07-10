import json

from transformers import LlamaTokenizer


class MiniCPMVTokenizer(LlamaTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.im_start = "<image>"
        self.im_end = "</image>"
        self.ref_start = "<ref>"
        self.ref_end = "</ref>"
        self.box_start = "<box>"
        self.box_end = "</box>"
        self.quad_start = "<quad>"
        self.quad_end = "</quad>"
        self.point_start = "<point>"
        self.point_end = "</point>"
        self.slice_start = "<slice>"
        self.slice_end = "</slice>"

    @property
    def eos_id(self):
        return self.sp_model.eos_id()

    @property
    def bos_id(self):
        return self.sp_model.bos_id()

    @property
    def unk_id(self):
        return self.sp_model.unk_id()

    @property
    def im_start_id(self):
        return self._convert_token_to_id(self.im_start)

    @property
    def im_end_id(self):
        return self._convert_token_to_id(self.im_end)

    def apply_chat_template(self, 
                            conversation, 
                            add_image_msg: bool=True):
        if isinstance(conversation, str):
            conversation = json.loads(conversation)
        
        prompt = ""
        for i, msg in enumerate(conversation):
            role = msg["role"]
            content = msg["content"]
            assert role in ["user", "assistant"]
            if i == 0:
                assert role == "user", "The role of first msg should be user"
                if add_image_msg is True and "(<image>./</image>)" not in content:
                    content = "(<image>./</image>)" + content
            prompt += "<用户>" if role == "user" else "<AI>"
            prompt += content
        prompt += "<AI>"
        return prompt
    