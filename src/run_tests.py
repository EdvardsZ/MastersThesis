from tests import *

def test_all():
    print("Running all tests...")
    print("---------------------------")
    print("Encoder tests:")
    print("1. Test conventional encoder")
    test_conventional_encoder()
    print("---------------------------")
    print("Decoder tests:")
    print("1. Test conventional decoder")
    #test_conventional_decoder()
    print("2. Test label conditioned decoder")
    #test_label_conditioned_decoder()
    return

if __name__ == "__main__":
    test_all()