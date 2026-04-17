3.5.1 Fine-tuning Methods
It has been argued that fine-tuning all of the model’s parameters can lead to suboptimal
results, particularly on low-resource tasks (Peters et al., 2019). Early results on transfer
learning for text classification tasks advocated fine-tuning only the parameters of a small
classifier that was fed sentence embeddings produced by a fixed pre-trained model (Subramanian et al., 2018; Kiros et al., 2015; Logeswaran and Lee, 2018; Hill et al., 2016; Conneau
et al., 2017). This approach is less applicable to our encoder-decoder model because the
entire decoder must be trained to output the target sequences for a given task. Instead, we
focus on two alternative fine-tuning approaches that update only a subset of the parameters
of our encoder-decoder model.
The first, “adapter layers” (Houlsby et al., 2019; Bapna et al., 2019), is motivated by
the goal of keeping most of the original model fixed while fine-tuning. Adapter layers are
additional dense-ReLU-dense blocks that are added after each of the preexisting feed-forward
networks in each block of the Transformer. These new feed-forward networks are designed
so that their output dimensionality matches their input. This allows them to be inserted
into the network with no additional changes to the structure or parameters. When finetuning, only the adapter layer and layer normalization parameters are updated. The main
hyperparameter of this approach is the inner dimensionality d of the feed-forward network,
which changes the number of new parameters added to the model. We experiment with
various values for d.
The second alternative fine-tuning method we consider is “gradual unfreezing” (Howard
and Ruder, 2018). In gradual unfreezing, more and more of the model’s parameters are finetuned over time. Gradual unfreezing was originally applied to a language model architecture
consisting of a single stack of layers. In this setting, at the start of fine-tuning only the
29
Raffel, Shazeer, Roberts, Lee, Narang, Matena, Zhou, Li and Liu
Fine-tuning method GLUE CNNDM SQuAD SGLUE EnDe EnFr EnRo
⋆ All parameters 83.28 19.24 80.88 71.36 26.98 39.82 27.65
Adapter layers, d = 32 80.52 15.08 79.32 60.40 13.84 17.88 15.54
Adapter layers, d = 128 81.51 16.62 79.47 63.03 19.83 27.50 22.63
Adapter layers, d = 512 81.54 17.78 79.18 64.30 23.45 33.98 25.81
Adapter layers, d = 2048 81.51 16.62 79.47 63.03 19.83 27.50 22.63
Gradual unfreezing 82.50 18.95 79.17 70.79 26.71 39.02 26.93
Table 10: Comparison of different alternative fine-tuning methods that only update a subset
of the model’s parameters. For adapter layers, d refers to the inner dimensionality
of the adapters.
parameters of the final layer are updated, then after training for a certain number of updates
the parameters of the second-to-last layer are also included, and so on until the entire
network’s parameters are being fine-tuned. To adapt this approach to our encoder-decoder
model, we gradually unfreeze layers in the encoder and decoder in parallel, starting from
the top in both cases. Since the parameters of our input embedding matrix and output
classification matrix are shared, we update them throughout fine-tuning. Recall that our
baseline model consists of 12 layers each in the encoder and decoder and is fine-tuned for
2
18 steps. As such, we subdivide the fine-tuning process into 12 episodes of 2
18/12 steps each
and train from layers 12 − n to 12 in the nth episode. We note that Howard and Ruder
(2018) suggested fine-tuning an additional layer after each epoch of training. However, since
our supervised data sets vary so much in size and since some of our downstream tasks are
actually mixtures of many tasks (GLUE and SuperGLUE), we instead adopt the simpler
strategy of fine-tuning an additional layer after every 2
18/12 steps.
A comparison of the performance of these fine-tuning approaches is shown in Table 10.
For adapter layers, we report the performance using an inner dimensionality d of 32, 128,
512, 2048. Pursuant with past results (Houlsby et al., 2019; Bapna et al., 2019) we find that
lower-resource tasks like SQuAD work well with a small value of d whereas higher resource
tasks require a large dimensionality to achieve reasonable performance. This suggests that
adapter layers could be a promising technique for fine-tuning on fewer parameters as long as
the dimensionality is scaled appropriately to the task size. Note that in our case we treat
GLUE and SuperGLUE each as a single “task” by concatenating their constituent data
sets, so although they comprise some low-resource data sets the combined data set is large
enough that it necessitates a large value of d. We found that gradual unfreezing caused
a minor degradation in performance across all tasks, though it did provide some speedup
during fine-tuning. Better results may be attainable by more carefully tuning the unfreezing
schedule.
