<p align="center">
<img src="img/Banner-MS-AI-Tour-26.png" alt="decorative banner" width="1200"/>
</p>

# [Microsoft AI Tour 2026](https://aitour.microsoft.com)

## LTG154: Breaking Down the AI Tour Advisor Agent

[![Microsoft Azure AI Foundry Discord](https://dcbadge.limes.pink/api/server/ByRwuEEgH4)](https://aka.ms/AIFoundryDiscord-AITour26)
[![Azure AI Foundry Developer Forum](https://img.shields.io/badge/GitHub-Azure_AI_Foundry_Developer_Forum-blue?style=for-the-badge&logo=github&color=adff2f&logoColor=fff)](https://aka.ms/AIFoundryForum-AITour26)

If you will be delivering this session, check the [session-delivery-sources](./session-delivery-resources/) folder for slides, scripts, and other resources.

### Session Description

Did you try out the AI Tour Advisor at the community booth?
In this talk, we’ll break down how the agent is built and dive into the technology used under the hood. We’ll cover everything from instructions and models to tools and function calling, and explore how the entire experience comes together through the interface and agent framework.


### 🧠 Learning Outcomes

By the end of this session, learners will be able to:
- Use models in Azure AI Foundry 
- How to build an agent in LangGraph and Chainlit   

### 💻 Technologies Used

1. Azure AI Foundry 
1. Chainlit
1. LangGraph

### 🔗 Session Resources
| Resources          | Links                             | Description        |
|:-------------------|:----------------------------------|:-------------------|
| Additional Resources for this session | [https://learn.microsoft.com](https://learn.microsoft.com?ocid=AITour26_resourcecenter_cnl) | Links specific to this session |

### 📚 Continued Learning Resources
| Resources          | Links                             | Description        |
|:-------------------|:----------------------------------|:-------------------|
| AI Tour 2026 Resource Center | [https://aka.ms/AITour26-Resource-Center](https://aka.ms/AITour26-Resource-Center?ocid=AITour26_resourcecenter_cnl) | Links to all repos for AI Tour 26 Sessions |
| Azure AI Foundry Community Discord | [![Microsoft Azure AI Foundry Discord](https://dcbadge.limes.pink/api/server/ByRwuEEgH4)](https://aka.ms/AIFoundryDiscord-AITour26)| Connect with the Azure AI Foundry Community! |
| Learn at AI Tour | [https://aka.ms/LearnAtAITour](https://aka.ms/LearnAtAITour?ocid=AITour26_resourcecenter_cnl) | Continue learning on Microsoft Learn |


## Content Owners

<table>
<tr>
    <td align="center"><a href="http://github.com/hnky">
        <img src="https://github.com/hnky.png" width="100px;" alt="Henk Boelman"
"/><br />
        <sub><b> Henk Boelman
</b></sub></a><br />
            <a href="https://github.com/hnky" title="talk">📢</a> 
    </td>
</tr></table>


## Responsible AI 

Microsoft is committed to helping our customers use our AI products responsibly, sharing our learnings, and building trust-based partnerships through tools like Transparency Notes and Impact Assessments. Many of these resources can be found at [https://aka.ms/RAI](https://aka.ms/RAI).
Microsoft’s approach to responsible AI is grounded in our AI principles of fairness, reliability and safety, privacy and security, inclusiveness, transparency, and accountability.

Large-scale natural language, image, and speech models - like the ones used in this sample - can potentially behave in ways that are unfair, unreliable, or offensive, in turn causing harms. Please consult the [Azure OpenAI service Transparency note](https://learn.microsoft.com/legal/cognitive-services/openai/transparency-note?tabs=text) to be informed about risks and limitations.

The recommended approach to mitigating these risks is to include a safety system in your architecture that can detect and prevent harmful behavior. [Azure AI Content Safety](https://learn.microsoft.com/azure/ai-services/content-safety/overview) provides an independent layer of protection, able to detect harmful user-generated and AI-generated content in applications and services. Azure AI Content Safety includes text and image APIs that allow you to detect material that is harmful. Within Azure AI Foundry portal, the Content Safety service allows you to view, explore and try out sample code for detecting harmful content across different modalities. The following [quickstart documentation](https://learn.microsoft.com/azure/ai-services/content-safety/quickstart-text?tabs=visual-studio%2Clinux&pivots=programming-language-rest) guides you through making requests to the service.

Another aspect to take into account is the overall application performance. With multi-modal and multi-models applications, we consider performance to mean that the system performs as you and your users expect, including not generating harmful outputs. It's important to assess the performance of your overall application using [Performance and Quality and Risk and Safety evaluators](https://learn.microsoft.com/azure/ai-studio/concepts/evaluation-metrics-built-in). You also have the ability to create and evaluate with [custom evaluators](https://learn.microsoft.com/azure/ai-studio/how-to/develop/evaluate-sdk#custom-evaluators).

You can evaluate your AI application in your development environment using the [Azure AI Evaluation SDK](https://microsoft.github.io/promptflow/index.html). Given either a test dataset or a target, your generative AI application generations are quantitatively measured with built-in evaluators or custom evaluators of your choice. To get started with the azure ai evaluation sdk to evaluate your system, you can follow the [quickstart guide](https://learn.microsoft.com/azure/ai-studio/how-to/develop/flow-evaluate-sdk). Once you execute an evaluation run, you can [visualize the results in Azure AI Foundry portal ](https://learn.microsoft.com/azure/ai-studio/how-to/evaluate-flow-results).
