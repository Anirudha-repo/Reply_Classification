1\) If you only had 200 labeled replies, how would you improve the model without collecting thousands more?



**I’d use transfer learning by fine-tuning a pre-trained model like DistilBERT so it can learn from a small dataset. I’d also create more examples through paraphrasing or controlled LLM prompts, and use active learning to label only the most useful extra data. To avoid overfitting, I’d rely on cross-validation and maybe combine multiple models.**



2\) How would you ensure your reply classifier doesn’t produce biased or unsafe outputs in production?



**First, I’d check the training data and model outputs for bias and monitor how well it performs across different types of inputs. I’d add simple filters for harmful or unsafe text and make sure low-confidence predictions are flagged or handled by a human. Over time, I’d retrain the model with corrected examples and keep track of fairness and accuracy**



3\) Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non-generic?



**I’d give the model clear context like who the recipient is, their company, and the purpose of the email. I’d also include a few good examples and set rules for tone, length, and avoiding clichés. Finally, I’d tune generation settings and filter out weak or repetitive outputs so the results are specific, personal, and professional.**

