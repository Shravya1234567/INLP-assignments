# Assignment-1

1. **tokenizer.py:** contains the code for tokenizing the input string.

    To run the code, use the following command:
    ```
    python3 tokenizer.py
    ```
    then prompt will ask for the input string, enter the string and press enter. The output will be the tokenized list of lists of tokens.

2. **lm.py:** contains the class for N_gram language model (with and without smoothing)

3. **scores.py:** calculates the perplexities on pride (test and train)  and ulysses (test and train) using LM1, LM2, LM3, LM4.

    To run the code, use the following command:
    ```
    python3 scores.py
    ```

4. **ulysses.pkl:** contains the saved models on ulysses data

5. **pride.pkl:** contains the saved models on pride data

6. **language_modelling.py:** Calculates perplexity of a sentence using the saved models on ulysses and pride data.

    To run the code, use the following command:
    ```
    python3 language_modelling.py <method> <corpus>
    ```
    where method is i, gt (i for interpolation and gt for good turing) and corpus is ulysses.txt or pride.txt.

    On running this command, the prompt will ask for the input sentence, enter the sentence and press enter. The output will be the perplexity of the sentence.

7. **generate.py:** Given a sentence, predicts the top k words that can follow the sentence.

    to run the code, use the following command:
    ```
    python3 generate.py <method> <corpus> <k>
    ```
    where method is i, gt (i for interpolation and gt for good turing) and corpus is ulysses.txt or pride.txt.

    On running this command, the prompt will ask for the input sentence, enter the sentence and press enter and then prompt will ask for the value of n (in the case of generation using N-gram model (without smoothing)). The output will be the top k words that can follow the sentence.

8. The files **2021101051_LM1_test-perplexity.txt, 2021101051_LM1_train-perplexity.txt, 2021101051_LM2_test-perplexity.txt, 2021101051_LM2_train-perplexity.txt, 2021101051_LM3_test-perplexity.txt, 2021101051_LM3_train-perplexity.txt, 2021101051_LM4_test-perplexity.txt, 2021101051_LM4_train-perplexity.txt** contain the perplexities of the test and train data of pride and ulysses using LM1, LM2, LM3, LM4.