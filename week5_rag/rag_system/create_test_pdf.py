from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_JUSTIFY

def create_test_pdf(filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=20,
        alignment=TA_JUSTIFY
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20,
        alignment=TA_JUSTIFY
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        leading=14
    )
    
    title = Paragraph("Machine Learning Fundamentals", title_style)
    story.append(title)
    story.append(Spacer(1, 0.2*inch))
    
    intro = Paragraph(
        "<b>Introduction to Machine Learning</b><br/><br/>"
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data and improve their performance without being explicitly programmed. "
        "It focuses on developing algorithms that can access data and use it to learn for themselves.",
        body_style
    )
    story.append(intro)
    
    key_concepts = Paragraph(
        "<b>Key Concepts</b><br/><br/>"
        "1. <b>Supervised Learning</b>: The algorithm learns from labeled training data and makes predictions about unseen data.<br/><br/>"
        "2. <b>Unsupervised Learning</b>: The algorithm finds patterns in unlabeled data without any prior training.<br/><br/>"
        "3. <b>Reinforcement Learning</b>: The algorithm learns through trial and error by receiving rewards or penalties.",
        body_style
    )
    story.append(key_concepts)
    
    nn_title = Paragraph("<b>Neural Networks</b>", heading_style)
    story.append(nn_title)
    
    nn_intro = Paragraph(
        "A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates.",
        body_style
    )
    story.append(nn_intro)
    
    arch = Paragraph(
        "<b>Architecture</b><br/><br/>"
        "- <b>Input Layer</b>: Receives the initial data<br/>"
        "- <b>Hidden Layers</b>: Process the data through weighted connections<br/>"
        "- <b>Output Layer</b>: Produces the final prediction or classification",
        body_style
    )
    story.append(arch)
    
    activation = Paragraph(
        "<b>Activation Functions</b><br/><br/>"
        "Common activation functions include:<br/>"
        "- <b>ReLU</b> (Rectified Linear Unit): f(x) = max(0, x)<br/>"
        "- <b>Sigmoid</b>: f(x) = 1 / (1 + e^(-x))<br/>"
        "- <b>Tanh</b>: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))",
        body_style
    )
    story.append(activation)
    
    story.append(PageBreak())
    
    dl_title = Paragraph("<b>Deep Learning</b>", heading_style)
    story.append(dl_title)
    
    dl_intro = Paragraph(
        "Deep learning is a subset of machine learning that uses multi-layered neural networks to learn from vast amounts of data. "
        "It has revolutionized fields such as computer vision, natural language processing, and speech recognition.",
        body_style
    )
    story.append(dl_intro)
    
    applications = Paragraph(
        "<b>Applications</b><br/><br/>"
        "1. <b>Computer Vision</b>: Image classification, object detection, facial recognition<br/><br/>"
        "2. <b>Natural Language Processing</b>: Machine translation, sentiment analysis, text generation<br/><br/>"
        "3. <b>Speech Recognition</b>: Voice assistants, transcription services<br/><br/>"
        "4. <b>Autonomous Vehicles</b>: Self-driving cars use deep learning for perception and decision-making",
        body_style
    )
    story.append(applications)
    
    training_title = Paragraph("<b>Training Neural Networks</b>", heading_style)
    story.append(training_title)
    
    process = Paragraph(
        "The process of training a neural network involves several key steps:",
        body_style
    )
    story.append(process)
    
    forward = Paragraph(
        "<b>Forward Propagation</b><br/><br/>"
        "Data flows through the network from input to output, with each layer applying weights and activation functions to transform the data.",
        body_style
    )
    story.append(forward)
    
    backprop = Paragraph(
        "<b>Backpropagation</b><br/><br/>"
        "The algorithm calculates the gradient of the loss function with respect to each weight by the chain rule, "
        "propagating errors backward from the output layer to the input layer.",
        body_style
    )
    story.append(backprop)
    
    optimization = Paragraph(
        "<b>Optimization</b><br/><br/>"
        "Gradient descent and its variants (SGD, Adam, RMSprop) are used to update weights and minimize the loss function.",
        body_style
    )
    story.append(optimization)
    
    story.append(PageBreak())
    
    reg_title = Paragraph("<b>Regularization Techniques</b>", heading_style)
    story.append(reg_title)
    
    reg_intro = Paragraph(
        "To prevent overfitting, several regularization techniques are employed:",
        body_style
    )
    story.append(reg_intro)
    
    techniques = Paragraph(
        "- <b>Dropout</b>: Randomly deactivates neurons during training<br/><br/>"
        "- <b>L1/L2 Regularization</b>: Adds penalty terms to the loss function<br/><br/>"
        "- <b>Early Stopping</b>: Halts training when validation performance degrades<br/><br/>"
        "- <b>Data Augmentation</b>: Increases training data diversity through transformations",
        body_style
    )
    story.append(techniques)
    
    eval_title = Paragraph("<b>Evaluation Metrics</b>", heading_style)
    story.append(eval_title)
    
    metrics = Paragraph(
        "Common metrics for evaluating machine learning models include:<br/><br/>"
        "- <b>Accuracy</b>: Proportion of correct predictions<br/><br/>"
        "- <b>Precision</b>: Proportion of true positives among predicted positives<br/><br/>"
        "- <b>Recall</b>: Proportion of true positives among actual positives<br/><br/>"
        "- <b>F1-Score</b>: Harmonic mean of precision and recall<br/><br/>"
        "- <b>ROC-AUC</b>: Area under the receiver operating characteristic curve",
        body_style
    )
    story.append(metrics)
    
    conclusion = Paragraph(
        "<b>Conclusion</b><br/><br/>"
        "Machine learning and deep learning continue to advance rapidly, enabling breakthroughs across numerous domains. "
        "Understanding these fundamental concepts is essential for anyone working in the field of artificial intelligence.",
        body_style
    )
    story.append(conclusion)
    
    doc.build(story)
    print(f"PDF created successfully: {filename}")

if __name__ == "__main__":
    create_test_pdf("test_en.pdf")
