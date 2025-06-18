from explainers.image.grad_cam import GradCAMExplainer
from explainers.image.integrated_gradients import IntegratedGradientsExplainer
from explainers.image.lime_image import LimeImageExplainer
from explainers.tabular.dice_explainer import DiCEExplainer
from explainers.tabular.lime_explainer import LIMEExplainer
from explainers.tabular.shap_explainer import SHAPExplainer
from explainers.text.attention_explainer import AttentionExplainer
from explainers.text.lime_text import LimeTextExplainer
from explainers.text.shap_text import SHAPTextExplainer


EXPLAINER_REGISTRY = {
    "grad_cam_image": GradCAMExplainer,
    "integrated_gradients_image": IntegratedGradientsExplainer,
    "lime_image": LimeImageExplainer,
    "dice_tabular": DiCEExplainer,
    "lime_tabular": LIMEExplainer,
    "shap_tabular": SHAPExplainer,
    "attention_text": AttentionExplainer,
    "lime_text": LimeTextExplainer,
    "shap_text": SHAPTextExplainer
}


def get_explainer(
    model,
    method: str,
    task_type: str,
    feature_names=None,
    **kwargs
):
    """
    通用解释器工厂函数
    :param model: 已加载的模型
    :param method: 解释方法名 (如 'grad_cam', 'lime', 'shap')
    :param task_type: 任务类型 ('classification', 'regression')
    :param feature_names: 特征名，可选
    :param kwargs: 其他参数，自动传递给解释器
    :return: 解释器实例
    """
    method = method.lower()
    if method not in EXPLAINER_REGISTRY:
        raise ValueError(f"不支持的解释方法: {method}，可用方法: {list(EXPLAINER_REGISTRY.keys())}")

    explainer_cls = EXPLAINER_REGISTRY[method]
    explainer_args = {}

    # 通用参数
    explainer_args['model'] = model
    explainer_args['task_type'] = task_type
    if feature_names is not None:
        explainer_args['feature_names'] = feature_names

    # 其他参数自动透传
    explainer_args.update(kwargs)

    explainer = explainer_cls(**explainer_args)
    return explainer

import torch
from torchvision.models import resnet18
from core.model_loader import ModelLoader
import types
def main():
    model = resnet18(pretrained=True)
    model.eval()
    # 2. 给模型添加predict方法
    def predict(self, x):
        with torch.no_grad():
            return self(x)
    model.predict = types.MethodType(predict, model)

    explainer = get_explainer(
        model=model,
        method="grad_cam_image",
        task_type="classification",
        target_layer = "layer4"
    )
    from PIL import Image
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt


    # 5. 读取和预处理图片
    img_path = '/data/duyongkun/CPX/classify/MLS-PJ/test_images/cat.png'
    input_image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,1], C,H,W
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(input_image)  # shape: (3, 224, 224)
    img_np = tensor.permute(1, 2, 0).numpy()  # HWC for explainer

    # ============ 关键：检查模型预测 ============
    # ResNet18 期望输入是 (N, 3, 224, 224)
    input_tensor = tensor.unsqueeze(0)  # shape: (1, 3, 224, 224)
    outputs = model.predict(input_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    top5_prob, top5_catid = torch.topk(probs, 5)

    print("Top-5 预测类别和概率：")
    for i in range(top5_prob.size(0)):
        print("类别ID: {}, 概率: {:.3f}".format(top5_catid[i].item(), top5_prob[i].item()))
    print("你设置的 target_class 是: {}".format(283))

    # 6. 生成Grad-CAM解释
    result = explainer.explain(img_np, target=283)

    # 7. 显示可视化结果
    superimposed = result.visualization['superimposed']
    plt.imshow(superimposed)
    plt.title('Grad-CAM Result')
    plt.axis('off')
    plt.show()

    # 选做：保存图片
    Image.fromarray(superimposed).save('gradcam_output.jpg')
    print("Grad-CAM 结果已保存为 gradcam_output.jpg")


if __name__ == '__main__':
    main()