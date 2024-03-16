# Aspect-Extraction

The goal of this qualification test is to use an Aspect Ex-
traction (AE) model to automatically identify and extract aspects or
features mentioned in a given text. As Aspect Extraction is a sub-task
of Aspect Based Sentiment Analysis (ABSA), After an extensive inves-
tigation on the matter, we determined that the SemEval-2014 Task 4
laptop reviews dataset was the most crucial one to employ [4], then we
chose the best performed pre-trained model called InstructABSA which
is an instruction learning paradigm for Aspect-Based Sentiment Analysis
(ABSA) subtasks.[6]. After running and evaluating the model on the la-
beled dataset, we used an unlabled dataset from amazon product reviews
for the cell phones and accessories category [1], moreover, we extracted
the aspects for this unlabeled dataset and exported the result as a csv
file
