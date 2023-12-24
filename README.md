# Iterative Text Editor

This project involves the fine-tuning of the T5 model on various tasks, such as
Grammar Error Correction (GEC), Formality Transfer, Coherence Enhancement, Simplicity Transformation, Paraphrasing,
Neutralization, and more. The model is designed to handle text modification instructions using a span corruption
approach, overcoming the limitation of T5's inability to generate long texts.

## Datasets Used

The fine-tuning process utilized the following datasets:
- C4 Dataset: Used for GEC task.
- FCE Dataset: Used for GEC task.
- Lang8 Dataset: Used for GEC task.
- Bea19 Dataset: Used for GEC task.
- GYAFC Dataset: Used for Formality Transfer.
- DiscoFuse Dataset: Used for enhancing text coherence.
- WikiAuto Dataset: Used for Simplicity task.
- WikiLarge Dataset: Used for Simplicity task.
- ParaBankV2 Dataset: Used for Paraphrasing task.
- WNC Dataset: Used for Neutralization task.
- APPDIA Dataset: Used for making text non-offensive.
- Paradetox Dataset: Used for making text non-toxic.
- IteraTeR Dataset: Used for tasks related to simplicity, coherence, and GEC.

## Fine-Tuning Approach

The model was fine-tuned using nearly 250 prompts for the specified tasks to enhance generalization. To address the
limitation of T5 in generating long texts, a span corruption approach was employed.  
The input text to the T5 encoder follows the format:  
&lt;Instruction&gt;: &lt;Input Text&gt; -> ..... &lt;Span&gt; .... &lt;Span&gt; ....  
where the decoder is responsible for filling in the spans. T5 special tokens <extra_token_id_0...> were used for
representing spans.

The fine-tuned model is available at [Hugging Face Repo](https://huggingface.co/reza-alipour/ft5).

