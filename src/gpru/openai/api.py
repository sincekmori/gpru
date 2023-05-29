"""Client implementation for the OpenAI API version `1.2.0`."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

from httpx._config import DEFAULT_TIMEOUT_CONFIG
from httpx._types import TimeoutTypes
from pydantic import BaseModel, Field

from gpru._client import kwargs_for_uploading, request_factory, stream_factory


class Error(BaseModel):
    message: Optional[str] = None
    type: Optional[str] = None
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: Error


class Logprobs(BaseModel):
    tokens: Optional[List[str]] = None
    token_logprobs: Optional[List[float]] = None
    top_logprobs: Optional[List[Dict[str, Any]]] = None
    text_offset: Optional[List[int]] = None


class Choice(BaseModel):
    text: Optional[str] = None
    index: Optional[int] = None
    logprobs: Optional[Logprobs] = None
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Completion(BaseModel):
    id: str
    object: str
    created: int
    model: str
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
    """The contents of the request to create completion."""

    model: str
    """
    ID of the model to use.

    Examples
    --------
    - `model="text-davinci-003"`
    """
    prompt: Optional[
        Union[str, List[str], List[int], List[List[int]]]
    ] = "<|endoftext|>"
    """
    The prompt(s) to generate completions for, encoded as a string, array of strings,
    array of tokens, or array of token arrays.

    Notes
    -----
    `"<|endoftext|>"` is the document separator that the model sees during training, so
    if a prompt is not specified the model will generate as if from the beginning of a
    new document.

    Examples
    --------
    - `prompt="This is a test."`
    - `prompt=["This is a test."]`
    - `prompt=[1212, 318, 257, 1332, 13]`
    - `prompt=[[1212, 318, 257, 1332, 13]]`
    """
    suffix: Optional[str] = None
    """
    The suffix that comes after a completion of inserted text.

    Examples
    --------
    - `suffix="test."`
    """
    max_tokens: Optional[int] = Field(16, ge=0)
    """
    The maximum number of [tokens](https://platform.openai.com/tokenizer) to generate in
    the completion.

    The token count of your prompt plus `max_tokens` cannot exceed the model's context
    length. Most models have a context length of 2048 tokens (except for the newest
    models, which support 4096).

    Examples
    --------
    - `max_tokens=16`
    """
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    """
    What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make
    the output more random, while lower values like 0.2 will make it more focused and
    deterministic.

    We generally recommend altering this or `top_p` but not both.

    Examples
    --------
    - `temperature=1.0`
    """
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    """
    An alternative to sampling with temperature, called nucleus sampling, where the
    model considers the results of the tokens with `top_p` probability mass. So 0.1
    means only the tokens comprising the top 10% probability mass are considered.

    We generally recommend altering this or `temperature` but not both.

    Examples
    --------
    - `top_p=1.0`
    """
    n: Optional[int] = Field(1, ge=1, le=128)
    """
    How many completions to generate for each prompt.

    Notes
    -----
    Because this parameter generates many completions, it can quickly consume your token
    quota. Use carefully and ensure that you have reasonable settings for `max_tokens`
    and `stop`.

    Examples
    --------
    - `n=1`
    """
    stream: Optional[bool] = False
    """Whether to stream back partial progress."""
    logprobs: Optional[int] = Field(None, ge=0, le=5)
    """
    Include the log probabilities on the `logprobs` most likely tokens, as well the
    chosen tokens.

    For example, if `logprobs` is 5, the API will return a list of the 5 most likely
    tokens. The API will always return the `logprob` of the sampled token, so there may
    be up to `logprobs+1` elements in the response. The maximum value for `logprobs` is
    5. If you need more than this, please contact us through our
    [Help center](https://help.openai.com)
    and describe your use case.
    """
    echo: Optional[bool] = False
    """Echo back the prompt in addition to the completion."""
    stop: Optional[Union[str, List[str]]] = None
    r"""
    Up to 4 sequences where the API will stop generating further tokens.

    The returned text will not contain the stop sequence.

    Examples
    --------
    - `stop="\n"`
    - `stop=["\n"]`
    """
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    """
    Number between -2.0 and 2.0. Positive values penalize new tokens based on whether
    they appear in the text so far, increasing the model's likelihood to talk about new
    topics.

    See Also
    --------
    [Parameter details](https://platform.openai.com/docs/api-reference/parameter-details)
    """
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    """
    Number between -2.0 and 2.0. Positive values penalize new tokens based on their
    existing frequency in the text so far, decreasing the model's likelihood to repeat
    the same line verbatim.

    See Also
    --------
    [Parameter details](https://platform.openai.com/docs/api-reference/parameter-details)
    """
    best_of: Optional[int] = Field(1, ge=0, le=20)
    """
    Generates `best_of` completions server-side and returns the "best" (the one with the
    highest log probability per token). Results cannot be streamed.

    When used with `n`, `best_of` controls the number of candidate completions and `n`
    specifies how many to return - `best_of` must be greater than `n`.

    Notes
    -----
    Because this parameter generates many completions, it can quickly consume your token
    quota. Use carefully and ensure that you have reasonable settings for `max_tokens`
    and `stop`.
    """
    logit_bias: Optional[Dict[str, Any]] = None
    """
    Modify the likelihood of specified tokens appearing in the completion. Accepts a
    json object that maps tokens (specified by their token ID in the GPT tokenizer) to
    an associated bias value from -100 to 100.

    You can use this [tokenizer tool](https://platform.openai.com/tokenizer) (which
    works for both GPT-2 and GPT-3) to convert text to token IDs. Mathematically,
    the bias is added to the logits generated by the model prior to sampling. The exact
    effect will vary per model, but values between -1 and 1 should decrease or increase
    likelihood of selection; values like -100 or 100 should result in a ban or exclusive
    selection of the relevant token.

    As an example, you can pass `{"50256": -100}` to prevent the `<|endoftext|>` token
    from being generated.
    """
    user: Optional[str] = None
    """
    A unique identifier representing your end-user, which can help OpenAI to monitor and
    detect abuse.

    See Also
    --------
    [End-user IDs](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids)

    Examples
    --------
    - `user="user-1234"`
    """


class Role(str, Enum):
    """The role of the author of this message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


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


class ChatCompletion(BaseModel):
    id: str
    object: str
    created: int
    model: str
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


class ChatCompletionRequest(BaseModel):
    model: str
    """
    ID of the model to use.

    See the [model endpoint compatibility table](https://platform.openai.com/docs/models/model-endpoint-compatibility)
    for details on which models work with the Chat API.
    """
    messages: List[Message] = Field(..., min_items=1)
    """
    The messages to generate chat completions for, in the chat format.

    See Also
    --------
    [Chat completions - Introduction](https://platform.openai.com/docs/guides/chat/introduction)
    """
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    """
    What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make
    the output more random, while lower values like 0.2 will make it more focused and
    deterministic.

    We generally recommend altering this or `top_p` but not both.
    """
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    """
    An alternative to sampling with temperature, called nucleus sampling, where the
    model considers the results of the tokens with `top_p` probability mass. So 0.1
    means only the tokens comprising the top 10% probability mass are considered.

    We generally recommend altering this or `temperature` but not both.
    """
    n: Optional[int] = Field(1, ge=1, le=128)
    """How many chat completion choices to generate for each input message."""
    stream: Optional[bool] = False
    """If set, partial message deltas will be sent, like in ChatGPT."""
    stop: Optional[Union[str, List[str]]] = None
    """Up to 4 sequences where the API will stop generating further tokens."""
    max_tokens: Optional[int] = None
    """
    The maximum number of tokens allowed for the generated answer.

    By default, the number of tokens the model can return will be (4096 - prompt
    tokens).
    """
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    """
    Number between -2.0 and 2.0. Positive values penalize new tokens based on whether
    they appear in the text so far, increasing the model's likelihood to talk about new
    topics.

    See Also
    --------
    [Parameter details](https://platform.openai.com/docs/api-reference/parameter-details)
    """
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    """
    Number between -2.0 and 2.0. Positive values penalize new tokens based on their
    existing frequency in the text so far, decreasing the model's likelihood to repeat
    the same line verbatim.

    See Also
    --------
    [Parameter details](https://platform.openai.com/docs/api-reference/parameter-details)
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
    """
    A unique identifier representing your end-user, which can help OpenAI to monitor and
    detect abuse.

    See Also
    --------
    [End-user IDs](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids)

    Examples
    --------
    - `user="user-1234"`
    """


class Edit(BaseModel):
    object: str
    created: int
    choices: List[Choice]
    usage: Usage

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
            Concatenated text.
        """
        return "".join(self.texts)


class EditRequest(BaseModel):
    model: str
    """
    ID of the model to use.

    You can use the `text-davinci-edit-001` or `code-davinci-edit-001` model with this
    endpoint.
    """
    input: Optional[str] = ""
    """The input text to use as a starting point for the edit."""
    instruction: str
    """The instruction that tells the model how to edit the prompt."""
    n: Optional[int] = Field(1, ge=1, le=20)
    """How many edits to generate for the input and instruction."""
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    """
    What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make
    the output more random, while lower values like 0.2 will make it more focused and
    deterministic.

    We generally recommend altering this or `top_p` but not both.
    """
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    """
    An alternative to sampling with temperature, called nucleus sampling, where the
    model considers the results of the tokens with top_p probability mass. So 0.1 means
    only the tokens comprising the top 10% probability mass are considered.

    We generally recommend altering this or `temperature` but not both.
    """


class ImageDatum(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None


class ImageList(BaseModel):
    created: int
    data: List[ImageDatum]


class ImageSize(str, Enum):
    """The size of the generated images."""

    SQUARE_256 = "256x256"
    SQUARE_512 = "512x512"
    SQUARE_1024 = "1024x1024"


class ImageFormat(str, Enum):
    """The format in which the generated images are returned."""

    URL = "url"
    B64_JSON = "b64_json"


class ImageRequest(BaseModel):
    prompt: str = Field(..., max_length=1000)
    """
    A text description of the desired image(s).

    The maximum length is 1000 characters.

    Examples
    --------
    - `prompt="A cute baby sea otter"`
    """
    n: Optional[int] = Field(1, ge=1, le=10)
    """
    The number of images to generate.

    Must be between 1 and 10.
    """
    size: Optional[ImageSize] = ImageSize.SQUARE_1024
    """The size of the generated images."""
    response_format: Optional[ImageFormat] = ImageFormat.URL
    """The format in which the generated images are returned."""
    user: Optional[str] = None
    """
    A unique identifier representing your end-user, which can help OpenAI to monitor and
    detect abuse.

    See Also
    --------
    [End-user IDs](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids)

    Examples
    --------
    - `user="user-1234"`
    """


class ImageEditing(BaseModel):
    image: Path
    """
    The image to edit.

    Must be a valid PNG file, less than 4MB, and square. If mask is not provided, image
    must have transparency, which will be used as the mask.
    """
    mask: Optional[Path] = None
    """
    An additional image whose fully transparent areas (e.g. where alpha is zero)
    indicate where `image` should be edited.

    Must be a valid PNG file, less than 4MB, and have the same dimensions as `image`.
    """
    prompt: str = Field(..., max_length=1000)
    """
    A text description of the desired image(s).

    The maximum length is 1000 characters.

    Examples
    --------
    - `prompt="A cute baby sea otter wearing a beret"`
    """
    n: Optional[int] = Field(1, ge=1, le=10)
    """
    The number of images to generate.

    Must be between 1 and 10.
    """
    size: Optional[ImageSize] = ImageSize.SQUARE_1024
    """The size of the generated images."""
    response_format: Optional[ImageFormat] = ImageFormat.URL
    """The format in which the generated images are returned."""
    user: Optional[str] = None
    """
    A unique identifier representing your end-user, which can help OpenAI to monitor and
    detect abuse.

    See Also
    --------
    [End-user IDs](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids)

    Examples
    --------
    - `user="user-1234"`
    """


class ImageVariation(BaseModel):
    image: Path
    """
    The image to use as the basis for the variation(s).

    Must be a valid PNG file, less than 4MB, and square.
    """
    n: Optional[int] = Field(1, ge=1, le=10)
    """
    The number of images to generate.

    Must be between 1 and 10.
    """
    size: Optional[ImageSize] = ImageSize.SQUARE_1024
    """The size of the generated images."""
    response_format: Optional[ImageFormat] = ImageFormat.URL
    """The format in which the generated images are returned."""
    user: Optional[str] = None
    """
    A unique identifier representing your end-user, which can help OpenAI to monitor and
    detect abuse.

    See Also
    --------
    [End-user IDs](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids)

    Examples
    --------
    - `user="user-1234"`
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
    model: str
    """ID of the model to use."""
    input: Union[str, List[str], List[int], List[List[int]]]
    """
    Input text to get embeddings for, encoded as a string or array of tokens.

    To get embeddings for multiple inputs in a single request, pass an array of strings
    or array of token arrays. Each input must not exceed 8192 tokens in length.

    Examples
    --------
    - `input="This is a test."`
    - `input=["This is a test."]`
    - `input=[1212, 318, 257, 1332, 13]`
    - `input=[[1212, 318, 257, 1332, 13]]`
    """
    user: Optional[str] = None
    """
    A unique identifier representing your end-user, which can help OpenAI to monitor and
    detect abuse.

    See Also
    --------
    [End-user IDs](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids)

    Examples
    --------
    - `user="user-1234"`
    """


class TranscriptionJson(BaseModel):
    text: str


class Segment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    transient: bool


class TranscriptionVerboseJson(BaseModel):
    task: str
    language: str
    duration: float
    segments: List[Segment]
    text: str


class TranscriptionFormat(str, Enum):
    """The format of the transcript output."""

    JSON = "json"
    TEXT = "text"
    SRT = "srt"
    VERBOSE_JSON = "verbose_json"
    VTT = "vtt"


class TranscriptionRequest(BaseModel):
    file: Path
    """The audio file to transcribe, in one of these formats: mp3, mp4, mpeg, mpga, m4a,
    wav, or webm.
    """
    model: str
    """
    ID of the model to use.

    Only `whisper-1` is currently available.
    """
    prompt: Optional[str] = None
    """
    An optional text to guide the model's style or continue a previous audio segment.

    The [prompt](https://platform.openai.com/docs/guides/speech-to-text/prompting)
    should match the audio language.
    """
    response_format: Optional[TranscriptionFormat] = TranscriptionFormat.JSON
    """The format of the transcript output."""
    temperature: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    """
    The sampling temperature, between 0 and 1.

    Higher values like 0.8 will make the output more random, while lower values like 0.2
    will make it more focused and deterministic. If set to 0, the model will use
    [log probability](https://en.wikipedia.org/wiki/Log_probability) to automatically
    increase the temperature until certain thresholds are hit.
    """
    language: Optional[str] = None
    """
    The language of the input audio.

    Supplying the input language in [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
    format will improve accuracy and latency.
    """


class TranslationRequest(BaseModel):
    file: Path
    """The audio file to translate, in one of these formats: mp3, mp4, mpeg, mpga, m4a,
    wav, or webm.
    """
    model: str
    """
    ID of the model to use.

    Only `whisper-1` is currently available.
    """
    prompt: Optional[str] = None
    """
    An optional text to guide the model's style or continue a previous audio segment.

    The [prompt](https://platform.openai.com/docs/guides/speech-to-text/prompting)
    should be in English.
    """
    response_format: Optional[TranscriptionFormat] = TranscriptionFormat.JSON
    """The format of the transcript output."""
    temperature: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    """
    The sampling temperature, between 0 and 1.

    Higher values like 0.8 will make the output more random, while lower values like 0.2
    will make it more focused and deterministic. If set to 0, the model will use
    [log probability](https://en.wikipedia.org/wiki/Log_probability) to automatically
    increase the temperature until certain thresholds are hit.
    """


class File(BaseModel):
    id: str
    object: str
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: Optional[str] = None
    status_details: Optional[Dict[str, Any]] = None


class FileList(BaseModel):
    object: str
    data: List[File]

    @property
    def ids(self) -> List[str]:
        """
        List of file IDs.

        Returns
        -------
        List[str]
            List of file IDs.
        """
        return [d.id for d in self.data]


class DeletionResult(BaseModel):
    id: str
    object: str
    deleted: bool


class FineTuneEvent(BaseModel):
    object: str
    created_at: int
    level: str
    message: str


class FineTuneEventList(BaseModel):
    object: str
    data: List[FineTuneEvent]


class FineTune(BaseModel):
    id: str
    object: str
    created_at: int
    updated_at: int
    model: str
    fine_tuned_model: Optional[str] = None
    organization_id: str
    status: str
    hyperparams: Dict[str, Any]
    training_files: List[File]
    validation_files: List[File]
    result_files: List[File]
    events: Optional[List[FineTuneEvent]] = None


class FineTuneList(BaseModel):
    object: str
    data: List[FineTune]

    @property
    def ids(self) -> List[str]:
        """
        List of fine-tune job IDs.

        Returns
        -------
        List[str]
            List of fine-tune job IDs.
        """
        return [d.id for d in self.data]


class FineTuneRequest(BaseModel):
    training_file: str
    """
    The ID of an uploaded file that contains training data.

    See [upload file](https://platform.openai.com/docs/api-reference/files/upload) for
    how to upload a file.

    Your dataset must be formatted as a JSONL file, where each training
    example is a JSON object with the keys "prompt" and "completion".
    Additionally, you must upload your file with the purpose `fine-tune`.

    See Also
    --------
    [fine-tuning guide](https://platform.openai.com/docs/guides/fine-tuning/creating-training-data)
    """
    validation_file: Optional[str] = None
    """
    The ID of an uploaded file that contains validation data.

    If you provide this file, the data is used to generate validation
    metrics periodically during fine-tuning. These metrics can be viewed in the
    [fine-tuning results file](https://platform.openai.com/docs/guides/fine-tuning/analyzing-your-fine-tuned-model)
    Your train and validation data should be mutually exclusive.

    Your dataset must be formatted as a JSONL file, where each validation
    example is a JSON object with the keys "prompt" and "completion".
    Additionally, you must upload your file with the purpose `fine-tune`.

    See Also
    --------
    [fine-tuning guide](https://platform.openai.com/docs/guides/fine-tuning/creating-training-data)
    """
    model: Optional[str] = "curie"
    """
    The name of the base model to fine-tune.

    You can select one of "ada", "babbage", "curie", "davinci", or a fine-tuned model
    created after 2022-04-21.

    See Also
    --------
    [Models](https://platform.openai.com/docs/models)
    """
    n_epochs: Optional[int] = 4
    """
    The number of epochs to train the model for.

    An epoch refers to one full cycle through the training dataset.
    """
    batch_size: Optional[int] = None
    """
    The batch size to use for training. The batch size is the number of training
    examples used to train a single forward and backward pass.

    By default, the batch size will be dynamically configured to be ~0.2% of the number
    of examples in the training set, capped at 256 - in general, we've found that larger
    batch sizes tend to work better for larger datasets.
    """
    learning_rate_multiplier: Optional[float] = None
    """
    The learning rate multiplier to use for training. The fine-tuning learning rate is
    the original learning rate used for pretraining multiplied by this value.

    By default, the learning rate multiplier is the 0.05, 0.1, or 0.2 depending on final
    `batch_size` (larger learning rates tend to perform better with larger batch sizes).
    We recommend experimenting with values in the range 0.02 to 0.2 to see what produces
    the best results.
    """
    prompt_loss_weight: Optional[float] = 0.01
    """
    The weight to use for loss on the prompt tokens. This controls how much the model
    tries to learn to generate the prompt (as compared to the completion which always
    has a weight of 1.0), and can add a stabilizing effect to training when completions
    are short.

    If prompts are extremely long (relative to completions), it may make sense to reduce
    this weight so as to avoid over-prioritizing learning the prompt.
    """
    compute_classification_metrics: Optional[bool] = False
    """
    If set, we calculate classification-specific metrics such as accuracy and F-1 score
    using the validation set at the end of every epoch.

    These metrics can be viewed in the [results file](https://platform.openai.com/docs/guides/fine-tuning/analyzing-your-fine-tuned-model).

    In order to compute classification metrics, you must provide a `validation_file`.
    Additionally, you must specify `classification_n_classes` for multiclass
    classification or `classification_positive_class` for binary classification.
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
    If this is provided, we calculate F-beta scores at the specified beta values. The
    F-beta score is a generalization of F-1 score. This is only used for binary
    classification.

    With a beta of 1 (i.e. the F-1 score), precision and recall are given the same
    weight. A larger beta score puts more weight on recall and less on precision. A
    smaller beta score puts more weight on precision and less on recall.
    """
    suffix: Optional[str] = Field(None, min_length=1, max_length=40)
    """
    A string of up to 40 characters that will be added to your fine-tuned model name.

    For example, a `suffix` of "custom-model-name" would produce a model name like
    `ada:ft-your-org:custom-model-name-2022-02-15-04-21-04`.
    """


class ModelPermission(BaseModel):
    id: str
    object: str
    created: int
    allow_create_engine: bool
    allow_sampling: bool
    allow_logprobs: bool
    allow_search_indices: bool
    allow_view: bool
    allow_fine_tuning: bool
    organization: str
    group: Optional[str]
    is_blocking: bool


class Model(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str
    permission: List[ModelPermission]
    root: str
    parent: Optional[str]


class ModelList(BaseModel):
    object: str
    data: List[Model]

    @property
    def ids(self) -> List[str]:
        """
        List of model IDs.

        Returns
        -------
        List[str]
            List of model IDs.
        """
        return [d.id for d in self.data]


class Categories(BaseModel):
    hate: bool
    hate_threatening: bool = Field(..., alias="hate/threatening")
    self_harm: bool = Field(..., alias="self-harm")
    sexual: bool
    sexual_minors: bool = Field(..., alias="sexual/minors")
    violence: bool
    violence_graphic: bool = Field(..., alias="violence/graphic")


class CategoryScores(BaseModel):
    hate: float
    hate_threatening: float = Field(..., alias="hate/threatening")
    self_harm: float = Field(..., alias="self-harm")
    sexual: float
    sexual_minors: float = Field(..., alias="sexual/minors")
    violence: float
    violence_graphic: float = Field(..., alias="violence/graphic")


class ModerationResult(BaseModel):
    flagged: bool
    """Set to `True` if the model classifies the content as violating OpenAI's usage
    policies, `False` otherwise.
    """
    categories: Categories
    """
    Contains a dictionary of per-category binary usage policies violation flags.

    For each category, the value is `True` if the model flags the corresponding category
    as violated, `False` otherwise.
    """
    category_scores: CategoryScores
    """
    Contains a dictionary of per-category raw scores output by the model, denoting the
    model's confidence that the input violates the OpenAI's policy for the category.

    The value is between 0 and 1, where higher values denote higher confidence. The
    scores should not be interpreted as probabilities.
    """


class Moderation(BaseModel):
    id: str
    model: str
    results: List[ModerationResult]


class ModerationRequest(BaseModel):
    input: Union[str, List[str]]
    """The input text to classify."""
    model: Optional[str] = "text-moderation-latest"
    """
    Two content moderations models are available: `text-moderation-stable` and `text-
    moderation-latest`.

    The default is `text-moderation-latest` which will be automatically upgraded over
    time. This ensures you are always using our most accurate model. If you use `text-
    moderation-stable`, we will provide advanced notice before updating the model.
    Accuracy of `text-moderation-stable` may be slightly lower than for `text-
    moderation-latest`.
    """


class OpenAiApi:
    """
    Client implementation for the OpenAI API version `1.2.0`.

    See Also
    --------
    [Official API specification](https://github.com/openai/openai-openapi/blob/master/openapi.yaml)

    Notes
    -----
    Clients for deprecated APIs are not implemented.
    """

    def __init__(
        self,
        key: str,
        organization_id: Optional[str] = None,
        timeout: Optional[TimeoutTypes] = DEFAULT_TIMEOUT_CONFIG,
        base_url: Optional[str] = "https://api.openai.com/v1",
    ) -> None:
        """
        Initialize a client for the OpenAI API version `1.2.0`.

        Parameters
        ----------
        key : str
            API key for authentication, which can be found on the [API Keys page](https://platform.openai.com/account/api-keys).
        organization_id : Optional[str]
            Optional value used to specify which organization's subscription quota is
            counted for API requests by users belonging to multiple organizations, which
            can be found on the [Organization's settings page](https://platform.openai.com/account/org-settings).
        timeout : Optional[TimeoutTypes]
            Timeout configuration to use when sending requests, by default
            `DEFAULT_TIMEOUT_CONFIG`. See [HTTPX - Timeout Configuration](https://www.python-httpx.org/advanced/#timeout-configuration)
            for more information.
        base_url : Optional[str]
            API endpoint, by default `"https://api.openai.com/v1"`.

        Examples
        --------
        >>> import os
        >>> key = os.environ["OPENAI_API_KEY"]
        >>> api = OpenAiApi(key)
        """
        client_kwargs = {
            "headers": self._build_base_headers(key, organization_id),
            "http2": True,
            "base_url": base_url,
            "timeout": timeout,
        }
        self._request = request_factory(ErrorResponse, **client_kwargs)
        self._stream = stream_factory(ErrorResponse, **client_kwargs)

    def _build_base_headers(
        self, key: str, organization_id: Optional[str]
    ) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {key}"}
        if organization_id is not None:
            headers["OpenAI-Organization"] = organization_id
        return headers

    def create_completion(
        self, completion_request: CompletionRequest
    ) -> Union[Generator[Completion, None, None], Completion]:
        """
        Create a completion for the provided prompt and parameters.

        Parameters
        ----------
        completion_request : CompletionRequest
            Specification of the completion to create.

        Returns
        -------
        Union[Generator[Completion, None, None], Completion]:
            Generator that streams the `Completion` if `completion_request.stream` is
            `True`, or created `Completion` instance otherwise.

        See Also
        --------
        [Official API reference](https://platform.openai.com/docs/api-reference/completions/create)
        """
        kwargs = {
            "method": "POST",
            "path": "/completions",
            "json": completion_request.dict(exclude_none=True),
        }

        if completion_request.stream is True:
            return self._stream(Completion, **kwargs)  # type: ignore[return-value]

        response = self._request(**kwargs)
        return Completion.parse_obj(response.json())

    def create_chat_completion(
        self, chat_completion_request: ChatCompletionRequest
    ) -> Union[Generator[ChatCompletion, None, None], ChatCompletion]:
        """
        Create a completion for the chat message.

        Parameters
        ----------
        chat_completion_request : ChatCompletionRequest
            Specification of the chat completion to create.

        Returns
        -------
        Union[Generator[ChatCompletion, None, None], ChatCompletion]
            Generator that streams the `ChatCompletion` if
            `chat_completion_request.stream` is `True`, or created `ChatCompletion`
            instance otherwise.

        See Also
        --------
        [Official API reference](https://platform.openai.com/docs/api-reference/chat/create)
        """
        kwargs = {
            "method": "POST",
            "path": "/chat/completions",
            "json": chat_completion_request.dict(exclude_none=True),
        }

        if chat_completion_request.stream is True:
            return self._stream(ChatCompletion, **kwargs)  # type: ignore[return-value]

        response = self._request(**kwargs)
        return ChatCompletion.parse_obj(response.json())

    def create_edit(self, edit_request: EditRequest) -> Edit:
        """
        Create a new edit for the provided input, instruction, and parameters.

        Parameters
        ----------
        edit_request : EditRequest
            Specification of the edit to create.

        Returns
        -------
        Edit
            Created `Edit` instance.

        See Also
        --------
        [Official API reference](https://platform.openai.com/docs/api-reference/edits/create)
        """
        response = self._request(
            "POST", "/edits", json=edit_request.dict(exclude_none=True)
        )
        return Edit.parse_obj(response.json())

    def create_images(self, image_request: ImageRequest) -> ImageList:
        """
        Create images given a prompt.

        Parameters
        ----------
        image_request: ImageRequest
            Specification of the images to create.

        Returns
        -------
        ImageList
            Created `ImageList` instance.

        See Also
        --------
        [Official API reference](https://platform.openai.com/docs/api-reference/images/create)
        """
        response = self._request(
            "POST", "/images/generations", json=image_request.dict(exclude_none=True)
        )
        return ImageList.parse_obj(response.json())

    def edit_image(self, image_editing: ImageEditing) -> ImageList:
        """
        Create edited or extended images given an original image and a prompt.

        Parameters
        ----------
        image_editing : ImageEditing
            Specification of the images to create.

        Returns
        -------
        ImageList
            Edited or extended `ImageList` instance.

        See Also
        --------
        [Official API reference](https://platform.openai.com/docs/api-reference/images/create-edit)
        """
        response = self._request(
            "POST", "/images/edits", **kwargs_for_uploading(image_editing)
        )
        return ImageList.parse_obj(response.json())

    def create_image_variation(self, image_variation: ImageVariation) -> ImageList:
        """
        Create a variation of a given image.

        Parameters
        ----------
        image_variation : ImageVariation
            Specification of the images to create.

        Returns
        -------
        ImageList
            Created `ImageList` instance.

        See Also
        --------
        [Official API reference](https://platform.openai.com/docs/api-reference/images/create-variation)
        """
        response = self._request(
            "POST", "/images/variations", **kwargs_for_uploading(image_variation)
        )
        return ImageList.parse_obj(response.json())

    def create_embedding(self, embedding_request: EmbeddingRequest) -> Embedding:
        """
        Create an embedding vector representing the input text.

        Parameters
        ----------
        embedding_request : EmbeddingRequest
            Specification of the embedding to create.

        Returns
        -------
        Embedding
            Created `Embedding` instance.

        See Also
        --------
        [Official API reference](https://platform.openai.com/docs/api-reference/embeddings/create)
        """
        response = self._request(
            "POST", "/embeddings", json=embedding_request.dict(exclude_none=True)
        )
        return Embedding.parse_obj(response.json())

    def transcribe_audio(
        self, transcription_request: TranscriptionRequest
    ) -> Union[TranscriptionJson, TranscriptionVerboseJson, str]:
        """
        Transcribe audio into the input language.

        Parameters
        ----------
        transcription_request : TranscriptionRequest
            Specification of the transcription to create.

        Returns
        -------
        Union[TranscriptionJson, TranscriptionVerboseJson, str]
            Created `TranscriptionJson` instance if
            `transcription_request.response_format` is `TranscriptionFormat.JSON`,
            created `TranscriptionVerboseJson` instance if
            `transcription_request.response_format` is
            `TranscriptionFormat.VERBOSE_JSON`, or `str` otherwise.

        See Also
        --------
        [Official API reference](https://platform.openai.com/docs/api-reference/audio/create)
        """
        response = self._request(
            "POST",
            "/audio/transcriptions",
            **kwargs_for_uploading(transcription_request),
        )

        if transcription_request.response_format is TranscriptionFormat.JSON:
            return TranscriptionJson.parse_obj(response.json())

        if transcription_request.response_format is TranscriptionFormat.VERBOSE_JSON:
            return TranscriptionVerboseJson.parse_obj(response.json())

        # text, srt or vtt
        return str(response.text)

    def translate_audio(
        self, translation_request: TranslationRequest
    ) -> Union[TranscriptionJson, TranscriptionVerboseJson, str]:
        """
        Translate audio into into English.

        Parameters
        ----------
        translation_request : TranslationRequest
            Specification of the transcription to create.

        Returns
        -------
        Union[TranscriptionJson, TranscriptionVerboseJson, str]
            Created `Edit` instance.

        See Also
        --------
        [Official API reference](https://platform.openai.com/docs/api-reference/audio/create)
        """
        response = self._request(
            "POST", "/audio/translations", **kwargs_for_uploading(translation_request)
        )

        if translation_request.response_format is TranscriptionFormat.JSON:
            return TranscriptionJson.parse_obj(response.json())

        if translation_request.response_format is TranscriptionFormat.VERBOSE_JSON:
            return TranscriptionVerboseJson.parse_obj(response.json())

        # text, srt or vtt
        return str(response.text)

    def list_files(self) -> FileList:
        """
        Return a list of files that belong to the user's organization.

        Returns
        -------
        FileList
            `FileList` instance.

        See Also
        --------
        [Official API reference](https://platform.openai.com/docs/api-reference/files/list)
        """
        response = self._request("GET", "/files")
        return FileList.parse_obj(response.json())

    def upload_file(self, file: Path, purpose: str) -> File:
        """
        Upload a file that contains document(s) to be used across various
        endpoints/features. Currently, the size of all the files uploaded by one
        organization can be up to 1 GB.

        Parameters
        ----------
        file : Path
            [JSON Lines](https://jsonlines.readthedocs.io/en/latest/) file to be
            uploaded. If the `purpose` is set to "fine-tune", each line is a JSON record
            with "prompt" and "completion" fields representing your
            [training examples](https://platform.openai.com/docs/guides/fine-tuning/prepare-training-data).
        purpose : str
            The intended purpose of the uploaded documents. Use "fine-tune" for
            [Fine-tuning](https://platform.openai.com/docs/api-reference/fine-tunes).
            This allows us to validate the format of the uploaded file.

        Returns
        -------
        File
            Uploaded `File` instance.

        See Also
        --------
        [Official API reference](https://platform.openai.com/docs/api-reference/files/upload)
        """
        data = {"purpose": purpose}
        files = {"file": file.open("rb")}
        response = self._request("POST", "/files", data=data, files=files)
        return File.parse_obj(response.json())

    def get_file(self, file_id: str) -> File:
        """
        Return information about a specific file.

        Parameters
        ----------
        file_id : str
            ID of the file.

        Returns
        -------
        File
            Specified `File` instance.

        See Also
        --------
        [Official API reference](https://platform.openai.com/docs/api-reference/files/retrieve)
        """
        response = self._request("GET", f"/files/{file_id}")
        return File.parse_obj(response.json())

    def delete_file(self, file_id: str) -> DeletionResult:
        """
        Delete a file.

        Parameters
        ----------
        file_id : str
            ID of the file.

        Returns
        -------
        DeletionResult
            `DeletionResult` instance.

        See Also
        --------
        [Official API reference](https://platform.openai.com/docs/api-reference/files/delete)
        """
        response = self._request("DELETE", f"/files/{file_id}")
        return DeletionResult.parse_obj(response.json())

    def download_file(self, file_id: str) -> str:
        """
        Return the contents of the specified file.

        Parameters
        ----------
        file_id : str
            ID of the file.

        Returns
        -------
        str
            Contents of the file.

        See Also
        --------
        [Official API reference](https://platform.openai.com/docs/api-reference/files/retrieve-content)
        """
        response = self._request("GET", f"/files/{file_id}/content")
        return str(response.text)

    def list_fine_tunes(self) -> FineTuneList:
        """
        List your organization's fine-tuning jobs.

        Returns
        -------
        FineTuneList
            `FineTuneList` instance.

        See Also
        --------
        [Official API reference](https://platform.openai.com/docs/api-reference/fine-tunes/list)
        """
        response = self._request("GET", "/fine-tunes")
        return FineTuneList.parse_obj(response.json())

    def create_fine_tune(self, fine_tune_request: FineTuneRequest) -> FineTune:
        """
        Create a job that fine-tunes a specified model from a given dataset.

        Response includes details of the enqueued job including job status and the name
        of the fine-tuned models once complete.

        Parameters
        ----------
        fine_tune_request : FineTuneRequest
            Specification of the fine-tune job to create.

        Returns
        -------
        FineTune
            Created `FineTune` instance.

        See Also
        --------
        - [Official API reference](https://platform.openai.com/docs/api-reference/fine-tunes/create)
        - [Learn more about Fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)
        """
        response = self._request(
            "POST", "/fine-tunes", json=fine_tune_request.dict(exclude_none=True)
        )
        return FineTune.parse_obj(response.json())

    def get_fine_tune(self, fine_tune_id: str) -> FineTune:
        """
        Get info about the fine-tune job.

        Parameters
        ----------
        fine_tune_id : str
            ID of the fine-tune job.

        Returns
        -------
        FineTune
            Specified `FineTune` instance.

        See Also
        --------
        - [Official API reference](https://platform.openai.com/docs/api-reference/fine-tunes/retrieve)
        - [Learn more about Fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)
        """
        response = self._request("GET", f"/fine-tunes/{fine_tune_id}")
        return FineTune.parse_obj(response.json())

    def cancel_fine_tune(self, fine_tune_id: str) -> FineTune:
        """
        Immediately cancel a fine-tune job.

        Parameters
        ----------
        fine_tune_id : str
            ID of the fine-tune job to cancel.

        Returns
        -------
        FineTune
            Canceled `FineTune` instance.

        See Also
        --------
        [Official API reference](https://platform.openai.com/docs/api-reference/fine-tunes/cancel)
        """
        response = self._request("POST", f"/fine-tunes/{fine_tune_id}/cancel")
        return FineTune.parse_obj(response.json())

    def list_fine_tune_events(
        self, fine_tune_id: str, stream: Optional[bool] = None
    ) -> Union[Generator[FineTuneEvent, None, None], FineTuneEventList]:
        """
        Get fine-grained status updates for a fine-tune job.

        Parameters
        ----------
        fine_tune_id : str
            ID of the fine-tune job to get events for.
        stream : Optional[bool], optional
            Whether to stream events for the fine-tune job.

        Returns
        -------
        Union[Generator[FineTuneEvent, None, None], FineTuneEventList]
            Generator that streams the `FineTuneEvent` if `stream` is `True`, or created
            `FineTuneEventList` instance otherwise.

        See Also
        --------
        [Official API reference](https://platform.openai.com/docs/api-reference/fine-tunes/events)
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

    def list_models(self) -> ModelList:
        """
        List the currently available models, and provides basic information about each
        one such as the owner and availability.

        Returns
        -------
        ModelList
            `ModelList` instance.

        See Also
        --------
        [Official API reference](https://platform.openai.com/docs/api-reference/models/list)
        """
        response = self._request("GET", "/models")
        return ModelList.parse_obj(response.json())

    def get_model(self, model_id: str) -> Model:
        """
        Retrieve a model instance, providing basic information about the model such as
        the owner and permissioning.

        Parameters
        ----------
        model_id : str
            ID of the model.

        Returns
        -------
        Model
            Specified `Model` instance.

        See Also
        --------
        [Official API reference](https://platform.openai.com/docs/api-reference/models/retrieve)
        """
        response = self._request("GET", f"/models/{model_id}")
        return Model.parse_obj(response.json())

    def delete_model(self, model_id: str) -> DeletionResult:
        """
        Delete a fine-tuned model. You must have the owner role in your organization.

        Parameters
        ----------
        model_id : str
            ID of the model to delete.

        Returns
        -------
        DeletionResult
            `DeletionResult` instance.

        See Also
        --------
        [Official API reference](https://platform.openai.com/docs/api-reference/fine-tunes/delete-model)
        """
        response = self._request("DELETE", f"/models/{model_id}")
        return DeletionResult.parse_obj(response.json())

    def create_moderation(self, moderation_request: ModerationRequest) -> Moderation:
        """
        Classify if text violates OpenAI's Content Policy.

        Notes
        -----
        OpenAI will continuously upgrade the moderation endpoint's underlying model.
        Therefore, custom policies that rely on `category_scores` may need recalibration
        over time.

        Parameters
        ----------
        moderation_request : ModerationRequest
            Specification of the moderation to create.

        Returns
        -------
        Moderation
            Created `Moderation` instance.

        See Also
        --------
        [Official API reference](https://platform.openai.com/docs/api-reference/moderations/create)
        """
        response = self._request(
            "POST", "/moderations", json=moderation_request.dict(exclude_none=True)
        )
        return Moderation.parse_obj(response.json())
