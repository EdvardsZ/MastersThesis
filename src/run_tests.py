from tests import test_conventional_encoder, test_conventional_decoder

def test_all():
    test_conventional_encoder()
    print("---------------------------")
    test_conventional_decoder()
    return

if __name__ == "__main__":
    test_all()