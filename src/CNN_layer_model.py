from typing import Optional, Union, Tuple, List

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaPreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.models.roberta.modeling_roberta import RobertaModel

class CNN_block(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CNN_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Transpose the input to match Conv1d input shape (batch_size, channels, sequence_length)
        x = x.transpose(1, 2)
        output = self.conv1(x)
        output = self.conv2(output)
        output = x + self.relu(output)
        # Transpose back to original shape (batch_size, sequence_length, channels)
        output = output.transpose(1, 2)
        output = self.layer_norm(output)
        return output

class CNN_RobertaForQuestionAnswering(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        
        self.cnn_block1 = CNN_block(config.hidden_size, config.hidden_size)
        self.cnn_block2 = CNN_block(config.hidden_size, config.hidden_size)
        self.cnn_block3 = CNN_block(config.hidden_size, config.hidden_size)
        self.cnn_block4 = CNN_block(config.hidden_size, config.hidden_size)
        self.cnn_block5 = CNN_block(config.hidden_size, config.hidden_size)
        
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
    
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        
        # Apply CNN layers
        sequence_output = self.cnn_block1(sequence_output)
        sequence_output = self.cnn_block2(sequence_output)
        sequence_output = self.cnn_block3(sequence_output)
        sequence_output = self.cnn_block4(sequence_output)
        sequence_output = self.cnn_block5(sequence_output)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )