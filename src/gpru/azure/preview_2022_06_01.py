"""Client implementation for the Azure OpenAI API version `2022-06-01-preview`."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

from httpx._config import DEFAULT_TIMEOUT_CONFIG
from httpx._types import TimeoutTypes
from pydantic import BaseModel, Field

from gpru.azure._api import Api
from gpru.azure._utils import deprecation_warning

deprecation_warning(__name__)


class ErrorCode(str, Enum):
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


class InnerErrorCode(str, Enum):
    """
    Inner error codes as defined in the Microsoft REST guidelines.

    See Also
    --------
    https://github.com/microsoft/api-guidelines/blob/vNext/Guidelines.md#7102-error-condition-responses
    """

    INVALID_PAYLOAD = "invalidPayload"
    """The request data is invalid for this operation."""


class InnerError(BaseModel):
    """
    Inner error as defined in the Microsoft REST guidelines.

    See Also
    --------
    https://github.com/microsoft/api-guidelines/blob/vNext/Guidelines.md#7102-error-condition-responses
    """

    code: Optional[InnerErrorCode] = None
    innererror: Optional["InnerError"] = None


class AuthoringError(BaseModel):
    """
    Error content as defined in the Microsoft REST guidelines.

    See Also
    --------
    https://github.com/microsoft/api-guidelines/blob/vNext/Guidelines.md#7102-error-condition-responses
    """

    code: ErrorCode
    message: str = Field(..., min_length=1)
    """The message of this error."""
    target: Optional[str] = None
    """The location where the error happened if available."""
    details: Optional[List["AuthoringError"]] = None
    """The error details if available."""
    innererror: Optional[InnerError] = None


class InferenceError(BaseModel):
    code: Optional[str] = None
    message: Optional[str] = None
    param: Optional[str] = None
    type: Optional[str] = None


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
    DEPLOYMENT = "deployment"
    """This object represents a deployment."""


class State(str, Enum):
    """The state of a job or item."""

    NOT_RUNNING = "notRunning"
    """The operation was created and is not queued to be processed in the future."""
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


class ScaleType(str, Enum):
    """Defines how scaling operations will be executed."""

    MANUAL = "manual"
    """Scaling of a deployment will happen by manually specifying the capacity of a
    model.
    """
    STANDARD = "standard"
    """Scaling of a deployment will happen automatically based on usage."""


class ScaleSettings(BaseModel):
    """
    The scale settings of a deployment.

    It defines the modes for scaling and the reserved capacity.
    """

    capacity: Optional[int] = None
    """The constant reserved capacity of the inference endpoint for this deployment."""
    scale_type: ScaleType


class Deployment(BaseModel):
    """Deployments manage the reserved quota for Azure OpenAI models and make them
    available for inference requests.
    """

    object: Optional[TypeDiscriminator] = None
    status: Optional[State] = None
    created_at: Optional[int] = None
    """A timestamp when this job or item was created (in unix epochs)."""
    updated_at: Optional[int] = None
    """A timestamp when this job or item was modified last (in unix epochs)."""
    id: Optional[str] = None
    """The identity of this item."""
    model: str = Field(..., min_length=1)
    """
    The OpenAI model to deploy.

    Can be a base model or a fine tune.
    """
    owner: Optional[str] = None
    """
    The owner of this deployment.

    For Azure OpenAI only "organization-owner" is supported.
    """
    scale_settings: ScaleSettings


class DeploymentList(BaseModel):
    """Represents a list of deployments."""

    object: Optional[TypeDiscriminator] = None
    data: Optional[List[Deployment]] = None
    """The list of items."""

    @property
    def ids(self) -> List[str]:
        """
        List of deployment IDs.

        Returns
        -------
        List[str]
            List of deployment IDs.
        """
        if self.data is None:
            return []
        return [d.id for d in self.data if d.id is not None]


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
    created_at: Optional[int] = None
    """A timestamp when this event was created (in unix epochs)."""
    level: Optional[LogLevel] = None
    message: Optional[str] = None
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
    """The identifier of the base model used for the fine-tune."""
    fine_tuned_model: Optional[str] = None
    """
    The identifier of the resulting fine tuned model.

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
    """The identifier of the base model used for this fine-tune."""
    training_file: str = Field(..., min_length=1)
    """The file that is used for training this fine tuned model."""
    validation_file: Optional[str] = None
    """The file that is used to evaluate the fine tuned model during training."""
    hyperparams: Optional[HyperParameters] = None


class Capabilities(BaseModel):
    """The capabilities of a base or fine tune model."""

    fine_tune: Optional[bool] = None
    """A value indicating whether a model can be used for fine tuning."""
    inference: Optional[bool] = None
    """A value indicating whether a model can be deployed."""
    completion: Optional[bool] = None
    """A value indicating whether a model supports completion."""
    embeddings: Optional[bool] = None
    """A value indicating whether a model supports embeddings."""
    scale_types: Optional[List[ScaleType]] = None
    """The supported scale types for deployments of this model."""


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
    inference: Optional[int] = None
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
    """The base model ID if this is a fine tune model; otherwise `None`."""
    fine_tune: Optional[str] = None
    """The fine tune job ID if this is a fine tune model; otherwise `None`."""
    capabilities: Optional[Capabilities] = None
    deprecation: Optional[Deprecation] = None


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


class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class Completion(BaseModel):
    id: Optional[str] = None
    object: Optional[str] = None
    created: Optional[int] = None
    model: Optional[str] = None
    choices: Optional[List[Choice]] = None
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
        if self.choices is None:
            return []
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
    prompt: Optional[Union[str, List[str], List[int], List[List[int]]]] = None
    """
    An optional prompt to complete from, encoded as a string, a list of strings, or a
    list of token lists.

    Defaults to <|endoftext|>. The prompt to complete from. If you would like to provide
    multiple prompts, use the POST variant of this method. Note that <|endoftext|> is
    the document separator that the model sees during training, so if a prompt is not
    specified the model will generate as if from the beginning of a new document.
    Maximum allowed size of string list is 2048.
    """
    max_tokens: Optional[int] = Field(16, ge=0)
    """
    The maximum number of tokens to generate.

    Has minimum of 0.
    """
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    """
    What sampling temperature to use.

    Higher values means the model will take more risks. Try 0.9 for more creative
    applications, and 0 (argmax sampling) for ones with a well-defined answer. We
    generally recommend using this or `top_p` but not both. Minimum of 0 and maximum of
    2 allowed.
    """
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    """
    An alternative to sampling with temperature, called nucleus sampling, where the
    model considers the results of the tokens with top_p probability mass.

    So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    We generally recommend using this or `temperature` but not both. Minimum of 0 and
    maximum of 1 allowed.
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
    """The ID of the end-user, for use in tracking and rate-limiting."""
    n: Optional[int] = Field(1, ge=1, le=128)
    """
    How many snippets to generate for each prompt.

    Minimum of 1 and maximum of 128 allowed.
    """
    stream: Optional[bool] = False
    """
    Whether to enable streaming for this endpoint.

    If set, tokens will be sent as server-sent events as they become available.
    """
    logprobs: Optional[int] = Field(None, ge=0, le=100)
    """
    Include the log probabilities on the `logprobs` most likely tokens, as well the
    chosen tokens.

    So for example, if `logprobs` is 10, the API will return a list of the 10 most
    likely tokens. If `logprobs` is 0, only the chosen tokens will have logprobs
    returned. Minimum of 0 and maximum of 100 allowed.
    """
    model: Optional[str] = None
    """The name of the model to use."""
    echo: Optional[bool] = False
    """Echo back the prompt in addition to the completion."""
    stop: Optional[Union[str, List[str]]] = None
    """A sequence which indicates the end of the current document."""
    completion_config: Optional[str] = None
    cache_level: Optional[int] = None
    """Can be used to disable any server-side caching, 0=no cache, 1=prompt prefix
    enabled, 2=full cache.
    """
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    """
    How much to penalize new tokens based on their existing frequency in the text so
    far.

    Decreases the model's likelihood to repeat the same line verbatim. Has minimum of -2
    and maximum of 2.
    """
    frequency_penalty: Optional[float] = 0.0
    """
    How much to penalize new tokens based on whether they appear in the text so far.

    Increases the model's likelihood to talk about new topics.
    """
    best_of: Optional[int] = Field(None, le=128)
    """
    How many generations to create server side, and display only the best.

    Will not stream intermediate progress if best_of > 1. Has maximum value of 128.
    """


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str], List[int], List[List[int]]]
    """An input to embed, encoded as a string, a list of strings, or a list of token
    lists.
    """
    user: Optional[str] = None
    """The ID of the end-user, for use in tracking and rate-limiting."""
    input_type: Optional[str] = None
    """Input type of embedding search to use."""
    model: Optional[str] = None
    """ID of the model to use."""


class AzureOpenAiApi(Api):
    """
    Client implementation for the Azure OpenAI API version `2022-06-01-preview`.

    See Also
    --------
    - [Official authoring API specification](https://github.com/Azure/azure-rest-api-specs/blob/main/specification/cognitiveservices/data-plane/AzureOpenAI/authoring/preview/2022-06-01-preview/azureopenai.json)
    - [Official inference API specification](https://github.com/Azure/azure-rest-api-specs/blob/main/specification/cognitiveservices/data-plane/AzureOpenAI/inference/preview/2022-06-01-preview/inference.json)
    """

    def __init__(
        self,
        endpoint: str,
        key: Optional[str] = None,
        ad_token: Optional[str] = None,
        timeout: Optional[TimeoutTypes] = DEFAULT_TIMEOUT_CONFIG,
    ) -> None:
        """
        Initialize a client for the Azure OpenAI API version `2022-06-01-preview`.

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
            "2022-06-01-preview",
            key,
            ad_token,
            timeout,
        )

    def list_deployments(self) -> DeploymentList:
        """
        Get the list of deployments owned by the Azure OpenAI resource.

        Returns
        -------
        DeploymentList
            `DeploymentList` instance.
        """
        response = self._request("GET", "/deployments")
        return DeploymentList.parse_obj(response.json())

    def create_deployment(self, deployment: Deployment) -> Deployment:
        """
        Create a new deployment for the Azure OpenAI resource according to the given
        specification.

        Parameters
        ----------
        deployment : Deployment
            Specification of the deployment including the model to deploy and the scale
            settings.

        Returns
        -------
        Deployment
            Created `Deployment` instance.
        """
        response = self._request(
            "POST", "/deployments", json=deployment.dict(exclude_none=True)
        )
        return Deployment.parse_obj(response.json())

    def get_deployment(self, deployment_id: str) -> Deployment:
        """
        Get details for a single deployment specified by the given `deployment_id`.

        Parameters
        ----------
        deployment_id : str
            ID of the deployment.

        Returns
        -------
        Deployment
            Specified `Deployment` instance.
        """
        response = self._request("GET", f"/deployments/{deployment_id}")
        return Deployment.parse_obj(response.json())

    def update_deployment(
        self,
        deployment_id: str,
        model: Optional[str] = None,
        scale_settings: Optional[ScaleSettings] = None,
    ) -> Deployment:
        """
        Update the mutable details of the deployment with the given `deployment_id`.

        Parameters
        ----------
        deployment_id : str
            ID of the deployment.
        model : Optional[str]
            The new OpenAI model to be used for this deployment. Can be a base model or
            a fine tune.
        scale_settings : Optional[ScaleSettings]
            The scale settings of a deployment.

        Returns
        -------
        Deployment
            Updated `Deployment` instance.
        """
        data: Dict[str, Any] = {"model": model}
        if scale_settings is not None:
            data["scale_settings"] = scale_settings.dict(exclude_none=True)
        response = self._request(
            "PATCH",
            f"/deployments/{deployment_id}",
            headers={"Content-Type": "application/merge-patch+json"},
            data=data,
        )
        return Deployment.parse_obj(response.json())

    def delete_deployment(self, deployment_id: str) -> None:
        """
        Delete the deployment specified by the given `deployment_id`.

        Parameters
        ----------
        deployment_id : str
            ID of the deployment.
        """
        self._request("DELETE", f"/deployments/{deployment_id}")

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
        return FileList.parse_obj(response.json())

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
        return File.parse_obj(response.json())

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
        return File.parse_obj(response.json())

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
        return File.parse_obj(response.json())

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
        return FineTuneList.parse_obj(response.json())

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
            "POST", "/fine-tunes", json=fine_tune_request.dict(exclude_none=True)
        )
        return FineTune.parse_obj(response.json())

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
        return FineTune.parse_obj(response.json())

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
        return FineTuneEventList.parse_obj(response.json())

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
        return FineTune.parse_obj(response.json())

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
        return ModelList.parse_obj(response.json())

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
        return Model.parse_obj(response.json())

    def create_completion(
        self, deployment_id: str, completion_request: CompletionRequest
    ) -> Union[Generator[Completion, None, None], Completion]:
        """
        Create a completion from a chosen model.

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
            "json": completion_request.dict(exclude_none=True),
        }

        if completion_request.stream is True:
            return self._stream(Completion, **kwargs)  # type: ignore[return-value]

        response = self._request(**kwargs)
        return Completion.parse_obj(response.json())

    def create_embedding(
        self, deployment_id: str, embedding_request: EmbeddingRequest
    ) -> str:
        """
        Return the embeddings for a given prompt.

        Parameters
        ----------
        deployment_id : str
            Deployment id of the model which was deployed.
        embedding_request : EmbeddingRequest
            Specification of the embedding to create.

        Returns
        -------
        str
            Created embedding.
        """
        response = self._request(
            "POST",
            f"/deployments/{deployment_id}/embeddings",
            json=embedding_request.dict(exclude_none=True),
        )
        return str(response.text)
