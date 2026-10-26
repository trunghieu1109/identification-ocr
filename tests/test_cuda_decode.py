from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
np = pytest.importorskip('numpy')

from identification_demo.cuda_decode import decode_memories, install_cuda_decoder, translate_on_device


class ScriptedModel:
    def __init__(self, scripts):
        self.scripts = scripts
        self.step = 0
        self.cnn = lambda image: image
        self.transformer = SimpleNamespace(forward_encoder=lambda value: value,
                                           forward_decoder=self.decode)

    def eval(self):
        pass

    def decode(self, prefix, memory):
        result = torch.full((len(self.scripts), len(prefix), 9), -5., device=prefix.device)
        for index, script in enumerate(self.scripts):
            token = script[min(self.step, len(script)-1)]
            result[index, -1, token] = 3.5 + index*.1
        self.step += 1
        return result, memory


@pytest.mark.parametrize('scripts,limit', [
    ([[4,5,2]],128), ([[4,2,6,7],[5,6,7,2]],128),
    ([[2],[2]],128), ([[4,5,6]],0), ([[4,5,6]],3),
])
def test_decoder_matches_upstream_tokens_stopping_and_confidences(scripts,limit):
    translate = pytest.importorskip('vietocr.tool.translate').translate
    image = torch.zeros(len(scripts),3,32,64)
    expected, old_scores = translate(image,ScriptedModel(scripts),max_seq_length=limit)
    actual, new_scores = translate_on_device(image,ScriptedModel(scripts),max_seq_length=limit)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_allclose(new_scores,old_scores,rtol=0,atol=1e-12,equal_nan=True)


def test_optimized_predictor_never_installs_on_cpu():
    with pytest.raises(ValueError,match='CUDA'):
        install_cuda_decoder(SimpleNamespace(device='cpu'))


def test_joined_memories_preserve_original_widths_and_transformer_predictions():
    translate = pytest.importorskip('vietocr.tool.translate').translate
    LanguageTransformer = pytest.importorskip('vietocr.model.seqmodel.transformer').LanguageTransformer
    torch.manual_seed(42)
    language = LanguageTransformer(9,8,2,1,1,16,32,0,0).eval()
    with torch.no_grad():
        language.fc.bias[2] = -100  # Exercise all steps without early EOS.
    model = SimpleNamespace(eval=language.eval, cnn=lambda x:x.transpose(0,1),transformer=language)
    images = [torch.randn(1,3,8),torch.randn(2,7,8)]
    expected = [translate(image,model,max_seq_length=3) for image in images]
    with torch.no_grad():
        memories = [language.forward_encoder(image.transpose(0,1)) for image in images]
        actual = decode_memories(language,memories,max_seq_length=3)
    for (tokens,scores),(old_tokens,old_scores) in zip(actual,expected):
        np.testing.assert_array_equal(tokens,old_tokens)
        np.testing.assert_allclose(scores,old_scores,rtol=1e-5,atol=1e-7,equal_nan=True)


def test_joining_groups_preserves_each_groups_original_eos_confidence_interval():
    scripts = [[4,2,6,6],[5,6,7,2]]
    def decoder(target,memory,**kwargs):
        step = len(target)-1
        return torch.tensor([[[scripts[int(memory[0,i,0])][step]*10+step] for i in range(memory.shape[1])]])
    def logits(output):
        encoded = output[...,0].long()
        values = torch.zeros(*encoded.shape,9)
        return values.scatter(-1,(encoded//10).unsqueeze(-1),(2+(encoded%10).float()).unsqueeze(-1))
    language = SimpleNamespace(d_model=1,gen_nopeek_mask=lambda n:torch.zeros(n,n),
        embed_tgt=lambda tokens:tokens.unsqueeze(-1),pos_enc=lambda x:x,
        transformer=SimpleNamespace(decoder=decoder),fc=logits)
    result = decode_memories(language,[torch.zeros(2,1,1),torch.ones(3,1,1)])
    assert result[0][0].tolist() == [[1,4,2]]
    assert result[1][0].tolist() == [[1,5,6,7,2]]
    first = float(torch.softmax(torch.tensor([2.,0,0,0,0,0,0,0,0]),0)[0])
    np.testing.assert_allclose(result[0][1],[first],rtol=1e-6)
