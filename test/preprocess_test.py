__author__ = 'workbook'

import preprocess
import unittest


class PreprocessTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_cleaned_parse(self):
        d = preprocess.cleaned_parse(train="test.csv", dump_file=None)
        self.assertEquals("I've decided to convert a Windows Phone 7 app that fetches an "
                                "XML feed and then parses it to an asp.net web app, using Visual "
                                "Web Developer Express. I figure since the code already works for "
                                "WP7, it should be a matter of mostly copying and pasting it for "
                                "the C# code behind. That's the first line of code from my WP7 "
                                "app that fetches the XML feed, but I can't even get HttpWebRequest "
                                "to work in Visual Web Developer like that. Intellisense shows a "
                                "create and createdefault, but no CreateHttp like there was in "
                                "Windows Phone 7. I just need to figure out how to fetch the page, "
                                "I assume the parsing will be the same as on my phone app. Any help? "
                                "Thanks, Amanda", str(d[8][1]))
        self.assertEquals("A lot of frameworks use URL conventions like which is great, but if you "
                          "need any configuration beyond that, it's up to you to write your own routes. "
                          "How would you handle URLs like on the backend? (to list all of a user's friends) "
                          "I'm thinking that in the controller, something like this would be appropriate: "
                          "Then you would have the following map: I wanted to put the Friend class inside "
                          "the User class but apparently you can't do that in PHP so this is the best I "
                          "could come up with. Thoughts? What URL would use for editing your list of friends? "
                          "could work, but it doesn't seem appropriate, since you should never be editing "
                          "someone else's friend list. Would be a better choice? Where would you put the "
                          "corresponding code for that? In a friend controller, or a user controller, or "
                          "a specialized account controller? Bonus question: which do you prefer? or The answers: "
                          "So, what I've gathered from the answers is that if the \"thing\" is complicated "
                          "(like \"friends\") but doesn't have its own controller, you can give it one "
                          "without a model, or if it's not, you should stuff it in with whatever it's most "
                          "closely related to. Your URLs should not influence where you put your code. "
                          "Most people seem to think you should stick to whever possible, because it's "
                          "what people are familiar with. No one really commented on the extended class "
                          "aside from saying it's \"awkward\". Perhaps FriendList would have been a more "
                          "appropriate class in that case if I really wanted to separate it out. Thanks "
                          "for all the answers :)", str(d[16][1]))
        self.assertEqual("/controller/action/{id}", d[16][2][0])
        self.assertEquals("""class User {
    function index() {
        echo 'user index';
    }
}

class Friend extends User {
    function index($user_id) {
        echo 'friend index';
    }
}
""", str(d[16][2][2]))

    def test_sample_index(self):
        ridx = {"hello": [(1, True), (3, False)], "world": [(3, True), (5, False)]}
        sidx = preprocess.tag_sample_index(ridx)
        self.assertEquals({1: [("hello", True)], 3: [("world", True), ("hello", False)], 5: [("world", False)]}, sidx)

if __name__ == '__main__':
    unittest.main()