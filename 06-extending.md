# Extending {#extending}

This chapter gives instructions on how to extend [mlr3](https://mlr3.mlr-org.com) and its extension packages with custom objects.

The approach is always the same:

1. determine the base class you want to inherit from,
2. extend the class with your custom functionality,
3. test your implementation
4. (optionally) add new object to the respective [`Dictionary`](https://mlr3misc.mlr-org.com/reference/Dictionary.html).

The chapter [Create a new learner](#extending-learners) illustrates the steps needed to create a custom learner in [mlr3](https://mlr3.mlr-org.com).
