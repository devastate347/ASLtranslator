PATH = 'savedModel.txt'
model = ASLTranslator()
model.load_state_dict(torch.load(PATH))
model.eval()