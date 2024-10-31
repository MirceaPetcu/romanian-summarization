Here is my Bachelor's thesis, which presents different Artificial Intelligence approaches for automatic summarization of Romanian text.

## Abstract
The limited time at our disposal constrains our quantity of information that we can
read and assimilate in a limited amount of time. To solve this problem, we need to come
up with effective solutions to capture the most important information in the shortest time.
In this paper, we propose a text summarization method for Romanian language
based on Machine Learning. The main requirement in Machine Learning represents the
large amount of data required to train the models. Hence, we created a data set that
consists of the news and their summaries from online press publications, with which
we trained an architecture-based Transformer model, which we later refine with Direct
Preference Optimization, with respect to human preferences.
The problem with current language models is that the number of tokens that can
be processed at once is limited due to memory and computational power considerations.
Thus, it is often used to truncate the text to a specific number of tokens, losing information.
So, we propose a preprocessing method with TextRank to preserve those more
important sentences in the truncation process.