Contribute
==========

Being an open source project, TorchDrug welcomes all kinds of contribution to keep it growing and thriving. Everyone can contribute, and we value every contribution.

It is also an incentive to the developers if you can acknowledge TorchDrug whenever it helps you. You can refer the library in your blog or papers, or star the repository to say "thank you".

Typical Contributions
---------------------

Here are some typical contributions you can make. All of them are equally important to the project.

- New features
- Datasets and models
- Application solutions
- Documents and tutorials
- Bug fixes

Implementation is not always required. You can also write proposals and discuss them with the community.

If you are a new open source contributor, we recommend you to take a look at these [guidelines].

[guidelines]: http://www.contribution-guide.org/

Submit an Issue
---------------

In the following cases, you are encouraged to [submit an issue] on GitHub.

[submit an issue]: https://github.com/DeepGraphLearning/torchdrug/issues

- Issues or bug reports
- Feature requests

We strongly recommend you to search the issue board before you submit any new issue. Sometimes the same issue may have been solved and you can find the solution in a closed issue.

Be as concrete as possible. For issues or bug reports, this means you are supposed to include all relevant information, such as code snippets, logs and package versions. For feature requests, please clearly describe the motivation, and possible scenarios for the feature.

It is not recommended to ask open-ended questions, such as a research problem on GitHub. Please post them in [TorchDrug Google Group].

[TorchDrug Google Group]: https://github.com/DeepGraphLearning/torchdrug/

Make a pull request
-------------------

You can contribute code to the project through pull requests. To make a pull request, you need

- [Fork the TorchDrug repository]
- Clone it to your local computer
- Implement your code and test it
- Push it to your GitHub repository
- [Create a pull request] in TorchDrug

[Fork the TorchDrug repository]: https://github.com/DeepGraphLearning/torchdrug/fork
[Create a pull request]: https://github.com/DeepGraphLearning/torchdrug/pulls

If your forked repository is behind the latest version, you can create a pull request in your repository to pull the latest version from TorchDrug.

You can directly make a pull request for trivial changes. For non-trivial changes, it is better to discuss it in an issue before you implement the code.

Code Development
----------------

Here are a few steps you need to develop TorchDrug.

First, uninstall existing TorchDrug in your environment. Alternatively, you can switch to a new conda environment.

```bash
conda uninstall torchdrug
```

Clone the repository and setup it in `develop` mode.

```bash
git clone https://github.com/DeepGraphLearning/torchdrug/
cd torchdrug
python setup.py develop
```

Now any change in the local code base will be reflected in the code importing them.