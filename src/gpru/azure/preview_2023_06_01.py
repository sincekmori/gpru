"""Client implementation for the Azure OpenAI API version `2023-06-01-preview`."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

from httpx._config import DEFAULT_TIMEOUT_CONFIG
from httpx._types import TimeoutTypes
from pydantic import BaseModel, Field, validator

from gpru.azure._api import Api


class AuthoringErrorCode(str, Enum):
    """
    Error codes as defined in the Microsoft REST guidelines.

    See Also
    --------
    https://github.com/microsoft/api-guidelines/blob/vNext/Guidelines.md#7102-error-condition-responses
    """

    CONFLICT = "conflict"
    """The requested operation conflicts with the current resource state."""
    INVALID_PAYLOAD = "invalidPayload"
    """The request data is invalid for this operation."""
    FORBIDDEN = "forbidden"
    """The operation is forbidden for the current user/api key."""
    NOT_FOUND = "notFound"
    """The resource is not found."""
    UNEXPECTED_ENTITY_STATE = "unexpectedEntityState"
    """The operation cannot be executed in the current resource's state."""
    ITEM_DOES_ALREADY_EXIST = "itemDoesAlreadyExist"
    """The item does already exist."""
    SERVICE_UNAVAILABLE = "serviceUnavailable"
    """The service is currently not available."""
    INTERNAL_FAILURE = "internalFailure"
    """
    Internal error.

    Please retry.
    """
    QUOTA_EXCEEDED = "quotaExceeded"
    """Quota exceeded."""
    JSONL_VALIDATION_FAILED = "jsonlValidationFailed"
    """Validation of jsonl data failed."""
    FILE_IMPORT_FAILED = "fileImportFailed"
    """Import of file failed."""
    TOO_MANY_REQUESTS = "tooManyRequests"
    """
    Too many requests.

    Please retry later.
    """
    UNAUTHORIZED = "unauthorized"
    """The current user/api key is not authorized for the operation."""
    CONTENT_FILTER = "contentFilter"
    """Image generation failed as a result of our safety system."""


class AuthoringInnerErrorCode(str, Enum):
    """
    Inner error codes as defined in the Microsoft REST guidelines.

    See Also
    --------
    https://github.com/microsoft/api-guidelines/blob/vNext/Guidelines.md#7102-error-condition-responses
    """

    INVALID_PAYLOAD = "invalidPayload"
    """The request data is invalid for this operation."""


class AuthoringInnerError(BaseModel):
    """
    Inner error as defined in the Microsoft REST guidelines.

    See Also
    --------
    https://github.com/microsoft/api-guidelines/blob/vNext/Guidelines.md#7102-error-condition-responses
    """

    code: Optional[AuthoringInnerErrorCode] = None
    innererror: Optional["AuthoringInnerError"] = None


class AuthoringError(BaseModel):
    """
    Error content as defined in the Microsoft REST guidelines.

    See Also
    --------
    https://github.com/microsoft/api-guidelines/blob/vNext/Guidelines.md#7102-error-condition-responses
    """

    code: AuthoringErrorCode
    message: str = Field(..., min_length=1)
    """The message of this error."""
    target: Optional[str] = None
    """The location where the error happened if available."""
    details: Optional[List["AuthoringError"]] = None
    """The error details if available."""
    innererror: Optional[AuthoringInnerError] = None


class InferenceInnerErrorCode(str, Enum):
    """Error codes for the inner error object."""

    RESPONSIBLE_AI_POLICY_VIOLATION = "ResponsibleAIPolicyViolation"
    """The prompt violated one of more content filter rules."""


class Severity(str, Enum):
    SAFE = "safe"
    """General content or related content in generic or non-harmful contexts."""
    LOW = "low"
    """Harmful content at a low intensity and risk level."""
    MEDIUM = "medium"
    """Harmful content at a medium intensity and risk level."""
    HIGH = "high"
    """Harmful content at a high intensity and risk level."""


class ContentFilterResult(BaseModel):
    severity: Severity
    filtered: bool


class InferenceErrorBase(BaseModel):
    code: Optional[str] = None
    message: Optional[str] = None


class ContentFilterResults(BaseModel):
    """Information about the content filtering category (hate, sexual, violence,
    self_harm), if it has been detected, as well as the severity level (very_low, low,
    medium, high-scale that determines the intensity and risk level of harmful content)
    and if it has been filtered or not.
    """

    sexual: Optional[ContentFilterResult] = None
    violence: Optional[ContentFilterResult] = None
    hate: Optional[ContentFilterResult] = None
    self_harm: Optional[ContentFilterResult] = None
    error: Optional[InferenceErrorBase] = None


class InferenceInnerError(BaseModel):
    """Inner error with additional details."""

    code: Optional[InferenceInnerErrorCode] = None
    content_filter_results: Optional[ContentFilterResults] = None


class InferenceError(InferenceErrorBase):
    requestid: Optional[str] = None
    param: Optional[str] = None
    type: Optional[str] = None
    inner_error: Optional[InferenceInnerError] = None


class ErrorResponse(BaseModel):
    """
    Error response as defined in the Microsoft REST guidelines.

    See Also
    --------
    https://github.com/microsoft/api-guidelines/blob/vNext/Guidelines.md#7102-error-condition-responses
    """

    error: Union[AuthoringError, InferenceError, None]


class TypeDiscriminator(str, Enum):
    """Defines the type of an object."""

    LIST = "list"
    """This object represents a list of other objects."""
    FINE_TUNE = "fine-tune"
    """This object represents a fine tune job."""
    FILE = "file"
    """This object represents a file."""
    FINE_TUNE_EVENT = "fine-tune-event"
    """This object represents an event of a fine tune job."""
    MODEL = "model"
    """This object represents a model (can be a base models or fine tune job result)."""


class State(str, Enum):
    """The state of a job or item."""

    NOT_RUNNING = "notRunning"
    """The operation was created and is queued to be processed in the future."""
    RUNNING = "running"
    """The operation has started to be processed."""
    SUCCEEDED = "succeeded"
    """The operation has successfully be processed and is ready for consumption."""
    CANCELED = "canceled"
    """The operation has been canceled and is incomplete."""
    FAILED = "failed"
    """The operation has completed processing with a failure and cannot be further
    consumed.
    """
    DELETED = "deleted"
    """The entity has been deleted but may still be referenced by other entities
    predating the deletion.
    """


class Purpose(str, Enum):
    """
    The intended purpose of the uploaded documents.

    Use `FINE_TUNE` for fine-tuning. This allows us to validate the format of the
    uploaded file.
    """

    FINE_TUNE = "fine-tune"
    """This file contains training data for a fine tune job."""
    FINE_TUNE_RESULTS = "fine-tune-results"
    """This file contains the results of a fine tune job."""


class FileStatistics(BaseModel):
    """
    A file is a document usable for training and validation.

    It can also be a service generated document with result details.
    """

    tokens: Optional[int] = None
    """The number of tokens used in prompts and completions for files of kind "fine-
    tune" once validation of file content is complete.
    """
    examples: Optional[int] = None
    """The number of contained training examples in files of kind "fine-tune" once
    validation of file content is complete.
    """


class File(BaseModel):
    """
    A file is a document usable for training and validation.

    It can also be a service generated document with result details.
    """

    object: Optional[TypeDiscriminator] = None
    status: Optional[State] = None
    created_at: Optional[int] = None
    """A timestamp when this job or item was created (in unix epochs)."""
    updated_at: Optional[int] = None
    """A timestamp when this job or item was modified last (in unix epochs)."""
    id: Optional[str] = None
    """The identity of this item."""
    bytes: Optional[int] = None
    """
    The size of this file when available (can be None).

    File sizes larger than 2^53-1 are not supported to ensure compatibility with
    JavaScript integers.
    """
    purpose: Purpose
    filename: str = Field(..., min_length=1)
    """The name of the file."""
    statistics: Optional[FileStatistics] = None
    error: Optional[AuthoringError] = None


class FileList(BaseModel):
    """Represents a list of files."""

    object: Optional[TypeDiscriminator] = None
    data: Optional[List[File]] = None
    """The list of items."""

    @property
    def ids(self) -> List[str]:
        """
        List of file IDs.

        Returns
        -------
        List[str]
            List of file IDs.
        """
        if self.data is None:
            return []
        return [d.id for d in self.data if d.id is not None]


class LogLevel(str, Enum):
    """The verbosity level of an event."""

    INFO = "info"
    """This event is for information only."""
    WARNING = "warning"
    """This event represents a mitigated issue."""
    ERROR = "error"
    """This message represents a non recoverable issue."""


class FineTuneEvent(BaseModel):
    object: Optional[TypeDiscriminator] = None
    created_at: int
    """A timestamp when this event was created (in unix epochs)."""
    level: LogLevel
    message: str = Field(..., min_length=1)
    """
    The message describing the event.

    This can be a change of state, e.g., enqueued, started, failed or completed, or
    other events like uploaded results.
    """


class FineTuneEventList(BaseModel):
    """Represents a list of events."""

    object: Optional[TypeDiscriminator] = None
    data: Optional[List[FineTuneEvent]] = None
    """The list of items."""


class HyperParameters(BaseModel):
    """The hyper parameter settings used in a fine tune job."""

    batch_size: Optional[int] = None
    """
    The batch size to use for training.

    The batch size is the number of training examples used to train a single forward and
    backward pass. In general, we've found that larger batch sizes tend to work better
    for larger datasets. The default value as well as the maximum value for this
    property are specific to a base model.
    """
    learning_rate_multiplier: Optional[float] = None
    """
    The learning rate multiplier to use for training.

    The fine-tuning learning rate is the original learning rate used for pre-training
    multiplied by this value. Larger learning rates tend to perform better with larger
    batch sizes. We recommend experimenting with values in the range 0.02 to 0.2 to see
    what produces the best results.
    """
    n_epochs: Optional[int] = None
    """
    The number of epochs to train the model for.

    An epoch refers to one full cycle through the training dataset.
    """
    prompt_loss_weight: Optional[float] = None
    """
    The weight to use for loss on the prompt tokens.

    This controls how much the model tries to learn to generate the prompt (as compared
    to the completion which always has a weight of 1.0), and can add a stabilizing
    effect to training when completions are short. If prompts are extremely long
    (relative to completions), it may make sense to reduce this weight so as to avoid
    over-prioritizing learning the prompt.
    """
    compute_classification_metrics: Optional[bool] = None
    """
    A value indicating whether to compute classification metrics.

    If set, we calculate classification-specific metrics such as accuracy and F-1 score
    using the validation set at the end of every epoch. These metrics can be viewed in
    the results file. In order to compute classification metrics, you must provide a
    validation_file. Additionally, you must specify classification_n_classes for
    multiclass classification or classification_positive_class for binary
    classification.
    """
    classification_n_classes: Optional[int] = None
    """
    The number of classes in a classification task.

    This parameter is required for multiclass classification.
    """
    classification_positive_class: Optional[str] = None
    """
    The positive class in binary classification.

    This parameter is needed to generate precision, recall, and F1 metrics when doing
    binary classification.
    """
    classification_betas: Optional[List[float]] = None
    """
    The classification beta values.

    If this is provided, we calculate F-beta scores at the specified beta values. The
    F-beta score is a generalization of F-1 score. This is only used for binary
    classification. With a beta of 1 (i.e. the F-1 score), precision and recall are
    given the same weight. A larger beta score puts more weight on recall and less on
    precision. A smaller beta score puts more weight on precision and less on recall.
    """


class FineTune(BaseModel):
    """Fine tuning is a job to tailor a model to specific training data."""

    object: Optional[TypeDiscriminator] = None
    status: Optional[State] = None
    created_at: Optional[int] = None
    """A timestamp when this job or item was created (in unix epochs)."""
    updated_at: Optional[int] = None
    """A timestamp when this job or item was modified last (in unix epochs)."""
    id: Optional[str] = None
    """The identity of this item."""
    model: str = Field(..., min_length=1)
    """The identifier (model-id) of the base model used for the fine-tune."""
    fine_tuned_model: Optional[str] = None
    """
    The identifier (model-id) of the resulting fine tuned model.

    This property is only populated for successfully completed fine-tune runs. Use this
    identifier to create a deployment for inferencing.
    """
    training_files: List[File]
    """The files that are used for training the fine tuned model."""
    validation_files: Optional[List[File]] = None
    """The files that are used to evaluate the fine tuned model during training."""
    result_files: Optional[List[File]] = None
    """
    The result files containing training and evaluation metrics in csv format.

    The file is only available for successfully completed fine-tune runs.
    """
    events: Optional[List[FineTuneEvent]] = None
    """The events that show the progress of the fine-tune run including queued, running
    and completed.
    """
    organisation_id: Optional[str] = None
    """
    The organisation id of this fine tune job.

    Unused on Azure OpenAI; compatibility for OpenAI only.
    """
    user_id: Optional[str] = None
    """
    The user id of this fine tune job.

    Unused on Azure OpenAI; compatibility for OpenAI only.
    """
    hyperparams: Optional[HyperParameters] = None
    suffix: Optional[str] = None
    """The suffix used to identify the fine-tuned model."""
    error: Optional[AuthoringError] = None


class FineTuneList(BaseModel):
    """Represents a list of fine tunes."""

    object: Optional[TypeDiscriminator] = None
    data: Optional[List[FineTune]] = None
    """The list of items."""

    @property
    def ids(self) -> List[str]:
        """
        List of fine-tune job IDs.

        Returns
        -------
        List[str]
            List of fine-tune job IDs.
        """
        if self.data is None:
            return []
        return [d.id for d in self.data if d.id is not None]


class FineTuneRequest(BaseModel):
    """Defines the values of a fine tune job."""

    model: str = Field(..., min_length=1)
    """The identifier (model-id) of the base model used for this fine-tune."""
    training_file: str = Field(..., min_length=1)
    """The file identity (file-id) that is used for training this fine tuned model."""
    validation_file: Optional[str] = None
    """The file identity (file-id) that is used to evaluate the fine tuned model during
    training.
    """
    suffix: Optional[str] = None
    """
    The suffix used to identify the fine-tuned model.

    The suffix can contain up to 40 characters (a-z, A-Z, 0-9,- and _) that will be
    added to your fine-tuned model name.
    """
    n_epochs: Optional[int] = None
    """
    The number of epochs to train the model for.

    An epoch refers to one full cycle through the training dataset.
    """
    batch_size: Optional[int] = None
    """
    The batch size to use for training.

    The batch size is the number of training examples used to train a single forward and
    backward pass. In general, we've found that larger batch sizes tend to work better
    for larger datasets. The default value as well as the maximum value for this
    property are specific to a base model.
    """
    learning_rate_multiplier: Optional[float] = None
    """
    The learning rate multiplier to use for training.

    The fine-tuning learning rate is the original learning rate used for pre-training
    multiplied by this value. Larger learning rates tend to perform better with larger
    batch sizes. We recommend experimenting with values in the range 0.02 to 0.2 to see
    what produces the best results.
    """
    prompt_loss_weight: Optional[float] = None
    """
    The weight to use for loss on the prompt tokens.

    This controls how much the model tries to learn to generate the prompt (as compared
    to the completion which always has a weight of 1.0), and can add a stabilizing
    effect to training when completions are short. If prompts are extremely long
    (relative to completions), it may make sense to reduce this weight so as to avoid
    over-prioritizing learning the prompt.
    """
    compute_classification_metrics: Optional[bool] = None
    """
    A value indicating whether to compute classification metrics.

    If set, we calculate classification-specific metrics such as accuracy and F-1 score
    using the validation set at the end of every epoch. These metrics can be viewed in
    the results file. In order to compute classification metrics, you must provide a
    validation_file.Additionally, you must specify classification_n_classes for
    multiclass classification or classification_positive_class for binary
    classification.
    """
    classification_n_classes: Optional[int] = None
    """
    The number of classes in a classification task.

    This parameter is required for multiclass classification.
    """
    classification_positive_class: Optional[str] = None
    """
    The positive class in binary classification.

    This parameter is needed to generate precision, recall, and F1 metrics when doing
    binary classification.
    """
    classification_betas: Optional[List[float]] = None
    """
    The classification beta values.

    If this is provided, we calculate F-beta scores at the specified beta values. The
    F-beta score is a generalization of F-1 score. This is only used for binary
    classification. With a beta of 1 (i.e.the F-1 score), precision and recall are given
    the same weight. A larger beta score puts more weight on recall and less on
    precision. A smaller beta score puts more weight on precision and less on recall.
    """


class ImageOperation(BaseModel):
    """
    Defines response after creating an operation to generate images.

    It contains the status and the id of the operation.
    """

    status: State
    id: str
    """The Id of the asynchronous operation that can be used to query its status
    later.
    """


class ImageSize(str, Enum):
    """The desired size of the generated images."""

    SQUARE_256 = "256x256"
    SQUARE_512 = "512x512"
    SQUARE_1024 = "1024x1024"


class ImageRequest(BaseModel):
    """Defines the request to create an operation to generate images."""

    prompt: str = Field(..., min_length=1)
    """A text description of the desired image(s)."""
    n: Optional[int] = Field(1, ge=1)
    """The number of images to generate."""
    size: Optional[ImageSize] = ImageSize.SQUARE_1024
    user: Optional[str] = None
    """A unique identifier representing your end-user, which can help to monitor and
    detect abuse.
    """


class Capabilities(BaseModel):
    """The capabilities of a base or fine tune model."""

    fine_tune: bool
    """A value indicating whether a model can be used for fine tuning."""
    inference: bool
    """A value indicating whether a model can be deployed."""
    completion: bool
    """A value indicating whether a model supports completion."""
    chat_completion: bool
    """A value indicating whether a model supports chat completion."""
    embeddings: bool
    """A value indicating whether a model supports embeddings."""


class LifeCycleStatus(str, Enum):
    """
    The life cycle status of a model.

    Notes
    -----
    A model can be promoted from `PREVIEW` to `GENERALLY_AVAILABLE`, but never
    from `GENERALLY_AVAILABLE` to `PREVIEW`.
    """

    PREVIEW = "preview"
    """Model is in preview and covered by the service preview terms."""
    GENERALLY_AVAILABLE = "generally-available"
    """Model is generally available."""


class Deprecation(BaseModel):
    """
    Defines the dates of deprecation for the different use cases of a model.

    Usually base models support 1 year of fine tuning after creation. Inference is
    typically supported 2 years after creation of base or fine tuned models. The exact
    dates are specified in the properties.
    """

    fine_tune: Optional[int] = None
    """
    The end date of fine tune support of this model.

    Will be `None` for fine tune models.
    """
    inference: int
    """The end date of inference support of this model."""


class Model(BaseModel):
    """A model is either a base model or the result of a successful fine tune job."""

    object: Optional[TypeDiscriminator] = None
    status: Optional[State] = None
    created_at: Optional[int] = None
    """A timestamp when this job or item was created (in unix epochs)."""
    updated_at: Optional[int] = None
    """A timestamp when this job or item was modified last (in unix epochs)."""
    id: Optional[str] = None
    """The identity of this item."""
    model: Optional[str] = None
    """The base model identity (model-id) if this is a fine tune model; otherwise
    `None`.
    """
    fine_tune: Optional[str] = None
    """The fine tune job identity (fine-tune-id) if this is a fine tune model; otherwise
    `None`.
    """
    capabilities: Capabilities
    lifecycle_status: LifeCycleStatus
    deprecation: Deprecation


class ModelList(BaseModel):
    """Represents a list of models."""

    object: Optional[TypeDiscriminator] = None
    data: Optional[List[Model]] = None
    """The list of items."""

    @property
    def ids(self) -> List[str]:
        """
        List of model IDs.

        Returns
        -------
        List[str]
            List of model IDs.
        """
        if self.data is None:
            return []
        return [d.id for d in self.data if d.id is not None]


class ImageResult(BaseModel):
    """The image url if successful, and an error otherwise."""

    url: Optional[str] = None
    """The URL that provides temporary access to download the generated image."""
    error: Optional[AuthoringError] = None


class OperationResult(BaseModel):
    """The result of the operation if the operation succeeded."""

    created: int
    """A timestamp when this job or item was created (in unix epochs)."""
    data: List[ImageResult]
    """The result data of the operation, if successful."""


class Operation(BaseModel):
    """
    The operation response containing the id and the status of the operation.

    If successful, it contains data with the generated images and an error otherwise.
    """

    id: str
    """The ID of the operation."""
    created: int
    """A timestamp when this job or item was created (in unix epochs)."""
    expires: Optional[int] = None
    """A timestamp when this operation and its associated images expire and will be
    deleted (in unix epochs).
    """
    result: Optional[OperationResult] = None
    """The result of the operation if the operation succeeded."""
    status: State
    error: Optional[AuthoringError] = None


class PromptFilterResult(BaseModel):
    """Content filtering results for a single prompt in the request."""

    prompt_index: Optional[int] = None
    content_filter_results: Optional[ContentFilterResults] = None


class Logprobs(BaseModel):
    tokens: Optional[List[str]] = None
    token_logprobs: Optional[List[float]] = None
    top_logprobs: Optional[List[Dict[str, float]]] = None
    text_offset: Optional[List[int]] = None


class Choice(BaseModel):
    text: Optional[str] = None
    index: Optional[int] = None
    logprobs: Optional[Logprobs] = None
    finish_reason: Optional[str] = None
    content_filter_results: Optional[ContentFilterResults] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Completion(BaseModel):
    id: str
    object: str
    created: int
    model: str
    prompt_filter_results: Optional[List[PromptFilterResult]]
    """
    Content filtering results for zero or more prompts in the request.

    In a streaming request, results for different prompts may arrive at different times
    or in different orders.

    Notes
    -----
    `prompt_filter_results` is required in the API specification, but as of July 1,
    2023, `prompt_filter_results` is not included in API responses.
    """
    choices: List[Choice]
    usage: Optional[Usage] = None

    @property
    def texts(self) -> List[str]:
        """
        List of texts from the choices.

        Returns
        -------
        List[str]
            List of texts.
        """
        return [c.text for c in self.choices if c.text is not None]

    @property
    def text(self) -> str:
        """
        The concatenation of the generated texts.

        Returns
        -------
        str
            Concatenated texts.
        """
        return "".join(self.texts)


class CompletionRequest(BaseModel):
    prompt: Optional[Union[str, List[str]]] = None
    """
    The prompt(s) to generate completions for, encoded as a string or array of strings.

    Note that <|endoftext|> is the document separator that the model sees during
    training, so if a prompt is not specified the model will generate as if from the
    beginning of a new document. Maximum allowed size of string list is 2048.
    """
    max_tokens: Optional[int] = Field(16, ge=0)
    """
    The token count of your prompt plus max_tokens cannot exceed the model's context
    length.

    Most models have a context length of 2048 tokens (except for the newest models,
    which support 4096). Has minimum of 0.
    """
    temperature: Optional[float] = 1.0
    """
    What sampling temperature to use.

    Higher values means the model will take more risks. Try 0.9 for more creative
    applications, and 0 (argmax sampling) for ones with a well-defined answer. We
    generally recommend altering this or top_p but not both.
    """
    top_p: Optional[float] = 1.0
    """
    An alternative to sampling with temperature, called nucleus sampling, where the
    model considers the results of the tokens with top_p probability mass.

    So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    We generally recommend altering this or temperature but not both.
    """
    logit_bias: Optional[Dict[str, Any]] = None
    """
    Defaults to `None`.

    Modify the likelihood of specified tokens appearing in the completion. Accepts a
    json object that maps tokens (specified by their token ID in the GPT tokenizer) to
    an associated bias value from -100 to 100. You can use this tokenizer tool (which
    works for both GPT-2 and GPT-3) to convert text to token IDs. Mathematically, the
    bias is added to the logits generated by the model prior to sampling. The exact
    effect will vary per model, but values between -1 and 1 should decrease or increase
    likelihood of selection; values like -100 or 100 should result in a ban or exclusive
    selection of the relevant token. As an example, you can pass {"50256" &#58; -100} to
    prevent the <|endoftext|> token from being generated.
    """
    user: Optional[str] = None
    """A unique identifier representing your end-user, which can help monitoring and
    detecting abuse.
    """
    n: Optional[int] = Field(1, ge=1, le=128)
    """
    How many completions to generate for each prompt.

    Minimum of 1 and maximum of 128 allowed.
    Note: Because this parameter generates many completions, it can quickly consume your
    token quota. Use carefully and ensure that you have reasonable settings for
    max_tokens and stop.
    """
    stream: Optional[bool] = False
    """
    Whether to stream back partial progress.

    If set, tokens will be sent as data-only server-sent events as they become
    available, with the stream terminated by a data: [DONE] message.
    """
    logprobs: Optional[int] = Field(None, ge=0, le=5)
    """
    Include the log probabilities on the logprobs most likely tokens, as well the chosen
    tokens.

    For example, if logprobs is 5, the API will return a list of the 5 most likely
    tokens. The API will always return the logprob of the sampled token, so there may be
    up to logprobs+1 elements in the response. Minimum of 0 and maximum of 5 allowed.
    """
    suffix: Optional[str] = None
    """The suffix that comes after a completion of inserted text."""
    echo: Optional[bool] = False
    """Echo back the prompt in addition to the completion."""
    stop: Optional[Union[str, List[str]]] = None
    """
    Up to 4 sequences where the API will stop generating further tokens.

    The returned text will not contain the stop sequence.
    """
    completion_config: Optional[str] = None
    cache_level: Optional[int] = None
    """Can be used to disable any server-side caching, 0=no cache, 1=prompt prefix
    enabled, 2=full cache.
    """
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    """
    Number between -2.0 and 2.0.

    Positive values penalize new tokens based on whether they appear in the text so far,
    increasing the model's likelihood to talk about new topics.
    """
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    """
    Number between -2.0 and 2.0.

    Positive values penalize new tokens based on their existing frequency in the text so
    far, decreasing the model's likelihood to repeat the same line verbatim.
    """
    best_of: Optional[int] = Field(None, le=128)
    """
    Generates best_of completions server-side and returns the "best" (the one with the
    highest log probability per token).

    Results cannot be streamed.
    When used with n, best_of controls the number of candidate completions and n
    specifies how many to return - best_of must be greater than n.

    Notes
    -----
    Because this parameter generates many completions, it can quickly consume your token
    quota. Use carefully and ensure that you have reasonable settings for max_tokens and
    stop. Has maximum value of 128.
    """


class EmbeddingDatum(BaseModel):
    index: int
    object: str
    embedding: List[float]


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class Embedding(BaseModel):
    object: str
    model: str
    data: List[EmbeddingDatum]
    usage: EmbeddingUsage

    @property
    def embeddings(self) -> List[List[float]]:
        """
        List of embeddings.

        Returns
        -------
        List[List[float]]
            List of embeddings.
        """
        return [d.embedding for d in self.data]


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    r"""
    Input text to get embeddings for, encoded as a string.

    To get embeddings for multiple inputs in a single request, pass an array of strings.
    Each input must not exceed 2048 tokens in length. Unless you are embedding code, we
    suggest replacing newlines (\n) in your input with a single space, as we have
    observed inferior results when newlines are present.
    """
    user: Optional[str] = None
    """A unique identifier representing your end-user, which can help monitoring and
    detecting abuse.
    """
    input_type: Optional[str] = None
    """Input type of embedding search to use."""


class Role(str, Enum):
    """The role of the author of this message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatCompletionMessage(BaseModel):
    role: Role
    content: str


class ChatCompletionDeltaMessage(BaseModel):
    role: Optional[Role] = None
    """The role of the author of this message."""
    content: Optional[str] = None
    """The contents of the message."""


class ChatChoice(BaseModel):
    index: Optional[int] = None
    message: Optional[ChatCompletionMessage] = None
    delta: Optional[ChatCompletionDeltaMessage] = None
    finish_reason: Optional[str] = None
    content_filter_results: Optional[ContentFilterResults] = None


class ChatCompletion(BaseModel):
    id: str
    object: str
    created: int
    model: str
    prompt_filter_results: Optional[List[PromptFilterResult]]
    """
    Content filtering results for zero or more prompts in the request.

    In a streaming request, results for different prompts may arrive at different times
    or in different orders.

    Notes
    -----
    `prompt_filter_results` is required in the API specification, but as of July 1,
    2023, `prompt_filter_results` is not included in API responses.
    """
    choices: List[ChatChoice]
    usage: Optional[Usage] = None

    @property
    def messages(self) -> List[ChatCompletionMessage]:
        """
        List of messages from the choices.

        Returns
        -------
        List[ChatCompletionMessage]
            List of messages.
        """
        return [c.message for c in self.choices if c.message is not None]

    @property
    def contents(self) -> List[str]:
        """
        List of message contents from the choices.

        Returns
        -------
        List[str]
            List of message contents
        """
        return [m.content for m in self.messages]

    @property
    def content(self) -> str:
        """
        The concatenation of the generated message contents.

        Returns
        -------
        str
            Concatenated message contents.
        """
        return "".join(self.contents)

    @property
    def delta_content(self) -> str:
        """
        Difference in content when streaming is enabled.

        Returns
        -------
        str
            Difference in content.
        """
        content = ""
        for choice in self.choices:
            if choice.delta is None:
                continue
            if choice.delta.content is None:
                continue
            content += choice.delta.content
        return content


class Message(BaseModel):
    role: Role
    """The role of the author of this message."""
    content: str
    """The contents of the message."""
    name: Optional[str] = None
    """The name of the user in a multi-user chat."""

    @validator("role", always=True)
    def role_must_be_other_than_tool(cls, v: Role) -> Role:  # noqa: N805
        if v is Role.TOOL:
            msg = (
                "`Role.TOOL` is not allowed. "
                "If you want to specify `Role.TOOL`, Use `ExtensionMessage` instead."
            )
            raise ValueError(msg)
        return v


class SystemMessage(Message):
    """A system message."""

    def __init__(self, content: str):
        """
        Create a system message.

        Parameters
        ----------
        content : str
            The contents of the message.
        """
        super().__init__(role=Role.SYSTEM, content=content)


class UserMessage(Message):
    """A user message."""

    def __init__(self, content: str):
        """
        Create a user message.

        Parameters
        ----------
        content : str
            The contents of the message.
        """
        super().__init__(role=Role.USER, content=content)


class AssistantMessage(Message):
    """An assistant message."""

    def __init__(self, content: str):
        """
        Create an assistant message.

        Parameters
        ----------
        content : str
            The contents of the message.
        """
        super().__init__(role=Role.ASSISTANT, content=content)


class _ChatCompletionRequestCommon(BaseModel):
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    """
    What sampling temperature to use, between 0 and 2.

    Higher values like 0.8 will make the output more random, while lower values like 0.2
    will make it more focused and deterministic. We generally recommend altering this or
    `top_p` but not both.
    """
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    """
    An alternative to sampling with temperature, called nucleus sampling, where the
    model considers the results of the tokens with top_p probability mass.

    So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    We generally recommend altering this or `temperature` but not both.
    """
    stream: Optional[bool] = False
    """
    If set, partial message deltas will be sent, like in ChatGPT.

    Tokens will be sent as data-only server-sent events as they become available, with
    the stream terminated by a `data: [DONE]` message.
    """
    stop: Optional[Union[str, List[str]]] = None
    """Up to 4 sequences where the API will stop generating further tokens."""
    max_tokens: Optional[int] = 4096
    """
    The maximum number of tokens allowed for the generated answer.

    By default, the number of tokens the model can return will be `4096 - prompt tokens`
    """
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    """
    Number between -2.0 and 2.0.

    Positive values penalize new tokens based on whether they appear in the text so far,
    increasing the model's likelihood to talk about new topics.
    """
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    """
    Number between -2.0 and 2.0.

    Positive values penalize new tokens based on their existing frequency in the text so
    far, decreasing the model's likelihood to repeat the same line verbatim.
    """
    logit_bias: Optional[Dict[str, Any]] = None
    """
    Modify the likelihood of specified tokens appearing in the completion.

    Accepts a json object that maps tokens (specified by their token ID in the
    tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is
    added to the logits generated by the model prior to sampling. The exact effect will
    vary per model, but values between -1 and 1 should decrease or increase likelihood
    of selection; values like -100 or 100 should result in a ban or exclusive selection
    of the relevant token.
    """
    user: Optional[str] = None
    """A unique identifier representing your end-user, which can help Azure OpenAI to
    monitor and detect abuse.
    """


class ChatCompletionRequest(_ChatCompletionRequestCommon):
    messages: List[Message] = Field(..., min_length=1)
    """The messages to generate chat completions for, in the chat format."""
    n: Optional[int] = Field(1, ge=1, le=128)
    """How many chat completion choices to generate for each input message."""


class ExtensionMessage(BaseModel):
    """A chat message for extensions."""

    index: Optional[int] = None
    """The index of the message in the conversation."""
    role: Role
    """The role of the author of this message."""
    recipient: Optional[str] = None
    """
    The recipient of the message in the format of <namespace>.<operation>.

    Present if and only if the recipient is tool.
    """
    content: str
    """The contents of the message."""
    end_turn: Optional[bool] = None
    """Whether the message ends the turn."""


class ExtensionChatChoice(BaseModel):
    index: Optional[int] = None
    messages: Optional[List[ExtensionMessage]] = None
    """The list of messages returned by the service."""
    finish_reason: Optional[str] = None


class ExtensionChatCompletion(BaseModel):
    """The response of the extensions chat completions."""

    id: str
    object: str
    created: int
    model: str
    choices: Optional[List[ExtensionChatChoice]] = None
    usage: Optional[Usage] = None


class DataSource(BaseModel):
    """The data source to be used for the Azure OpenAI on your data feature."""

    type: str
    """The data source type."""
    parameters: Optional[Dict[str, Any]] = None
    """The parameters to be used for the data source in runtime."""


class ExtensionChatCompletionRequest(_ChatCompletionRequestCommon):
    """Request for the chat completions using extensions."""

    messages: List[ExtensionMessage]
    dataSources: Optional[List[DataSource]] = None  # noqa: N815
    """The data sources to be used for the Azure OpenAI on your data feature."""


class AzureOpenAiApi(Api):
    """
    Client implementation for the Azure OpenAI API version `2023-06-01-preview`.

    See Also
    --------
    - [Official authoring API specification](https://github.com/Azure/azure-rest-api-specs/blob/main/specification/cognitiveservices/data-plane/AzureOpenAI/authoring/preview/2023-06-01-preview/azureopenai.json)
    - [Official inference API specification](https://github.com/Azure/azure-rest-api-specs/blob/main/specification/cognitiveservices/data-plane/AzureOpenAI/inference/preview/2023-06-01-preview/inference.json)
    """

    def __init__(
        self,
        endpoint: str,
        key: Optional[str] = None,
        ad_token: Optional[str] = None,
        timeout: Optional[TimeoutTypes] = DEFAULT_TIMEOUT_CONFIG,
    ) -> None:
        """
        Initialize a client for the Azure OpenAI API version `2023-06-01-preview`.

        Notes
        -----
        Either `key` or `ad_token` is required.

        Parameters
        ----------
        endpoint : str
            URL in the form of `"https://{your-resource-name}.openai.azure.com/"`
        key : Optional[str]
            Secret key for the Azure OpenAI API.
        ad_token : Optional[str]
            Azure Active Directory token.
        timeout : Optional[TimeoutTypes]
            Timeout configuration to use when sending requests, by default
            `DEFAULT_TIMEOUT_CONFIG`. See [HTTPX - Timeout Configuration](https://www.python-httpx.org/advanced/#timeout-configuration)
            for more information.

        Raises
        ------
        InvalidConfigError
            When the configuration is invalid.

        Examples
        --------
        >>> import os
        >>> endpoint = os.environ["AZURE_OPENAI_API_ENDPOINT"]
        >>> key = os.environ["AZURE_OPENAI_API_KEY"]
        >>> api = AzureOpenAiApi(endpoint, key)
        """
        super().__init__(
            ErrorResponse,
            endpoint,
            "2023-06-01-preview",
            key,
            ad_token,
            timeout,
        )

    def list_files(self) -> FileList:
        """
        Get a list of all files owned by the Azure OpenAI resource. These include user
        uploaded content like files with `Purpose.FINE_TUNE` for training or validation
        of fine-tunes models as well as files that are generated by the service such as
        `Purpose.FINE_TUNE_RESULTS` which contains various metrics for the corresponding
        fine-tune job.

        Returns
        -------
        FileList
            `FileList` instance.
        """
        response = self._request("GET", "/files")
        return FileList.model_validate(response.json())

    def upload_file(self, file: Path, purpose: Purpose) -> File:
        """
        Create a new file entity by uploading data from a local machine. Uploaded files
        can, for example, be used for training or evaluating fine-tuned models.

        Parameters
        ----------
        file : Path
            Gets or sets the file to upload into Azure OpenAI.
        purpose : Purpose
            The intended purpose of the uploaded documents. Use `Purpose.FINE_TUNE` for
            fine-tuning. This allows us to validate the format of the uploaded file.

        Returns
        -------
        File
            Uploaded `File` instance.
        """
        data = {"purpose": purpose.value}
        files = {"file": file.open("rb")}
        response = self._request("POST", "/files", data=data, files=files)
        return File.model_validate(response.json())

    def get_file(self, file_id: str) -> File:
        """
        Get details for a single file specified by the given `file_id` including status,
        size, purpose, etc.

        Parameters
        ----------
        file_id : str
            ID of the file.

        Returns
        -------
        File
            Specified `File` instance.
        """
        response = self._request("GET", f"/files/{file_id}")
        return File.model_validate(response.json())

    def delete_file(self, file_id: str) -> None:
        """
        Delete the file with the given `file_id`. Deletion is also allowed if a file was
        used, e.g., as training file in a fine-tune job.

        Parameters
        ----------
        file_id : str
            ID of the file.
        """
        self._request("DELETE", f"/files/{file_id}")

    def download_file(self, file_id: str) -> str:
        """
        Get the content of the file specified by the given `file_id`. Files can be user
        uploaded content or generated by the service like result metrics of a fine-tune
        job.

        Parameters
        ----------
        file_id : str
            ID of the file.

        Returns
        -------
        str
            Contents of the file.
        """
        response = self._request("GET", f"/files/{file_id}/content")
        return str(response.text)

    def import_file(self, content_url: str, filename: str, purpose: Purpose) -> File:
        """
        Create a new file entity by importing data from a provided url. Uploaded files
        can, for example, be used for training or evaluating fine-tuned models.

        Parameters
        ----------
        content_url : str
            The url to download the document from (can be SAS url of a blob or any other
            external url accessible with a GET request).
        filename : str
            The name of the [JSON Lines](https://jsonlines.readthedocs.io/en/latest/)
            file to be uploaded. If the `purpose` is set to `Purpose.FINE_TUNE`, each
            line is a JSON record with "prompt" and "completion" fields representing
            your training examples.
        purpose : Purpose
            The intended purpose of the uploaded documents.

        Returns
        -------
        File
            Created `File` instance.
        """
        response = self._request(
            "POST",
            "/files/import",
            json={
                "content_url": content_url,
                "filename": filename,
                "purpose": purpose.value,
            },
        )
        return File.model_validate(response.json())

    def list_fine_tunes(self) -> FineTuneList:
        """
        Get a list of all fine-tune jobs owned by the Azure OpenAI resource. The details
        that are returned for each fine-tune job contain besides its identifier the base
        model, training and validation files, hyper parameters, time stamps, status and
        events. Events are created when the job status changes, e.g. running or
        complete, and when results are uploaded.

        Returns
        -------
        FineTuneList
            `FineTuneList` instance.
        """
        response = self._request("GET", "/fine-tunes")
        return FineTuneList.model_validate(response.json())

    def create_fine_tune(self, fine_tune_request: FineTuneRequest) -> FineTune:
        """
        Create a job that fine-tunes a specified model from a given training file.
        Response includes details of the enqueued job including job status and hyper
        parameters. The name of the fine-tuned model is added to the response once
        complete.

        Parameters
        ----------
        fine_tune_request : FineTuneRequest
            Specification of the fine-tuned model to create. Required parameters are the
            base model and the training file to use. Optionally a validation file can be
            specified to compute validation metrics during training. Hyper parameters
            will be set to default values or can by optionally specified. These include
            batch size, learning rate multiplier, number of epochs and others.

        Returns
        -------
        FineTune
            Created `FineTune` instance.
        """
        response = self._request(
            "POST", "/fine-tunes", json=fine_tune_request.model_dump(exclude_none=True)
        )
        return FineTune.model_validate(response.json())

    def get_fine_tune(self, fine_tune_id: str) -> FineTune:
        """
        Get details for a single fine-tune job specified by the given `fine_tune_id`.
        The details contain the base model, training and validation files, hyper
        parameters, time stamps, status and events. Events are created when the job
        status changes, e.g. running or complete, and when results are uploaded.

        Parameters
        ----------
        fine_tune_id : str
            ID of the fine-tune job.

        Returns
        -------
        FineTune
            Specified `FineTune` instance.
        """
        response = self._request("GET", f"/fine-tunes/{fine_tune_id}")
        return FineTune.model_validate(response.json())

    def delete_fine_tune(self, fine_tune_id: str) -> None:
        """
        Delete the fine-tune job specified by the given `fine_tune_id`.

        Parameters
        ----------
        fine_tune_id : str
            ID of the fine-tune job.
        """
        self._request("DELETE", f"/fine-tunes/{fine_tune_id}")

    def list_fine_tune_events(
        self, fine_tune_id: str, stream: Optional[bool] = None
    ) -> Union[Generator[FineTuneEvent, None, None], FineTuneEventList]:
        """
        Get the events for the fine-tune job specified by the given fine_tune_id. Events
        are created when the job status changes, e.g. running or complete, and when
        results are uploaded.

        Parameters
        ----------
        fine_tune_id : str
            ID of the fine-tune job.
        stream : bool
            A flag indicating whether to stream events for the fine-tune job.

        Returns
        -------
        Union[Generator[FineTuneEvent, None, None], FineTuneEventList]:
            Generator that streams the `FineTuneEvent` if `stream` is `True`, or created
            `FineTuneEventList` instance otherwise.
        """
        kwargs = {
            "method": "GET",
            "path": f"/fine-tunes/{fine_tune_id}/events",
            "params": None if stream is None else {"stream": stream},
        }

        if stream is True:
            return self._stream(FineTuneEvent, **kwargs)  # type: ignore[return-value]

        response = self._request(**kwargs)
        return FineTuneEventList.model_validate(response.json())

    def cancel_fine_tune(self, fine_tune_id: str) -> FineTune:
        """
        Cancel the processing of the fine-tune job specified by the given
        `fine_tune_id`.

        Parameters
        ----------
        fine_tune_id : str
            ID of the fine-tune job.

        Returns
        -------
        FineTune
            Canceled `FineTune` instance.
        """
        response = self._request("POST", f"/fine-tunes/{fine_tune_id}/cancel")
        return FineTune.model_validate(response.json())

    def create_images(self, image_request: ImageRequest) -> ImageOperation:
        """
        Generate a batch of images from a text caption.

        Parameters
        ----------
        image_request : ImageRequest
            The specification of the images that should be generated.

        Returns
        -------
        ImageOperation
            Created `ImageOperation` instance.
        """
        response = self._request(
            "POST",
            "/images/generations:submit",
            json=image_request.model_dump(exclude_none=True),
        )
        return ImageOperation.model_validate(response.json())

    def list_models(self) -> ModelList:
        """
        Get a list of all models that are accessible by the Azure OpenAI resource. These
        include base models as well as all successfully completed fine-tuned models
        owned by the Azure OpenAI resource.

        Returns
        -------
        ModelList
            `ModelList` instance.
        """
        response = self._request("GET", "/models")
        return ModelList.model_validate(response.json())

    def get_model(self, model_id: str) -> Model:
        """
        Get details for the model specified by the given `model_id`.

        Parameters
        ----------
        model_id : str
            ID of the model.

        Returns
        -------
        Model
            Specified `Model` instance.
        """
        response = self._request("GET", f"/models/{model_id}")
        return Model.model_validate(response.json())

    def get_image_operation(self, operation_id: str) -> Operation:
        """
        Return the status of the images operation.

        Parameters
        ----------
        operation_id : str
            ID of the operation.

        Returns
        -------
        Operation
            Specified `Operation` instance.
        """
        response = self._request("GET", f"/operations/images/{operation_id}")
        return Operation.model_validate(response.json())

    def delete_image_operation(self, operation_id: str) -> None:
        """
        Delete an operation (if in terminal state) and all generated and user provided
        images associated with the operation.

        Parameters
        ----------
        operation_id : str
            ID of the operation.
        """
        self._request("DELETE", f"/operations/images/{operation_id}")

    def create_completion(
        self, deployment_id: str, completion_request: CompletionRequest
    ) -> Union[Generator[Completion, None, None], Completion]:
        """
        Create a completion for the provided prompt, parameters and chosen model.

        Parameters
        ----------
        deployment_id : str
            Deployment id of the model which was deployed.
        completion_request : CompletionRequest
            Specification of the completion to create.

        Returns
        -------
        Union[Generator[Completion, None, None], Completion]:
            Generator that streams the `Completion` if `completion_request.stream` is
            `True`, or created `Completion` instance otherwise.
        """
        kwargs = {
            "method": "POST",
            "path": f"/deployments/{deployment_id}/completions",
            "json": completion_request.model_dump(exclude_none=True),
        }

        if completion_request.stream is True:
            return self._stream(Completion, **kwargs)  # type: ignore[return-value]

        response = self._request(**kwargs)
        return Completion.model_validate(response.json())

    def create_embedding(
        self, deployment_id: str, embedding_request: EmbeddingRequest
    ) -> Embedding:
        """
        Get a vector representation of a given input that can be easily consumed by
        machine learning models and algorithms.

        Parameters
        ----------
        deployment_id : str
            Deployment id of the model which was deployed.
        embedding_request : EmbeddingRequest
            Specification of the embedding to create.

        Returns
        -------
        Embedding
            Created `Embedding` instance.
        """
        response = self._request(
            "POST",
            f"/deployments/{deployment_id}/embeddings",
            json=embedding_request.model_dump(exclude_none=True),
        )
        return Embedding.model_validate(response.json())

    def create_chat_completion(
        self, deployment_id: str, chat_completion_request: ChatCompletionRequest
    ) -> Union[Generator[ChatCompletion, None, None], ChatCompletion]:
        """
        Create a completion for the chat message.

        Parameters
        ----------
        deployment_id : str
            Deployment id of the model which was deployed.
        chat_completion_request : ChatCompletionRequest
            Specification of the chat completion to create.

        Returns
        -------
        Union[Generator[ChatCompletion, None, None], ChatCompletion]:
            Generator that streams the `ChatCompletion` if
            `chat_completion_request.stream` is `True`, or created `ChatCompletion`
            instance otherwise.
        """
        kwargs = {
            "method": "POST",
            "path": f"/deployments/{deployment_id}/chat/completions",
            "json": chat_completion_request.model_dump(exclude_none=True),
        }

        if chat_completion_request.stream is True:
            return self._stream(ChatCompletion, **kwargs)  # type: ignore[return-value]

        response = self._request(**kwargs)
        return ChatCompletion.model_validate(response.json())

    def create_extension_chat_completion(
        self,
        deployment_id: str,
        extension_chat_completion_request: ExtensionChatCompletionRequest,
    ) -> Union[Generator[ExtensionChatCompletion, None, None], ExtensionChatCompletion]:
        """
        Create a completion for the chat messages using extensions.

        Parameters
        ----------
        deployment_id : str
            Deployment id of the model which was deployed.
        extension_chat_completion_request : ExtensionChatCompletionRequest
            Specification of the extension chat completion to create.

        Returns
        -------
        Union[Generator[ExtensionChatCompletion, None, None], ExtensionChatCompletion]:
            Generator that streams the `ExtensionChatCompletion` if
            `extension_chat_completion_request.stream` is `True`, or created
            `ExtensionChatCompletion` instance otherwise.
        """
        kwargs = {
            "method": "POST",
            "path": f"/deployments/{deployment_id}/extensions/chat/completions",
            "json": extension_chat_completion_request.model_dump(exclude_none=True),
        }

        if extension_chat_completion_request.stream is True:
            return self._stream(ExtensionChatCompletion, **kwargs)  # type: ignore[return-value]

        response = self._request(**kwargs)
        return ExtensionChatCompletion.model_validate(response.json())
